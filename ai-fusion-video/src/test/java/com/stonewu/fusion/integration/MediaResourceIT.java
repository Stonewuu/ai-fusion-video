package com.stonewu.fusion.integration;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.media.MediaStorageConfigGuard;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.storage.S3ClientFactory;
import com.stonewu.fusion.service.storage.S3StorageConfigResolver;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockMultipartFile;
import org.testcontainers.junit.jupiter.Testcontainers;
import java.io.ByteArrayInputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doThrow;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.head;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@Testcontainers(disabledWithoutDocker = false)
class MediaResourceIT extends MediaResourceTestContext {
    @Test
    void uploadsThroughHttpReplaysAndReadsAuthenticatedByteRanges() throws Exception {
        String requestId = id();
        var upload = new MockMultipartFile("file", "封面.png", "image/png", png);
        String response = http.perform(multipart("/api/media-resources/upload").file(upload)
                        .param("requestId", requestId).param("projectId", "71")).andExpect(status().isOk())
                .andReturn().getResponse().getContentAsString(StandardCharsets.UTF_8);
        String mediaId = mapper.readTree(response).path("data").path("id").asText();
        var media = queries.get(mediaId, OWNER);
        assertThat(media.content().kind()).isEqualTo(MediaKind.IMAGE);
        assertThat(media.content().width()).isEqualTo(16);
        assertThat(media.content().height()).isEqualTo(10);
        assertThat(media.state()).isEqualTo("READY");
        assertThat(response).doesNotContain("keyset", "storage_config", "objectKey", "ticket=");
        assertThat(upload(requestId, png, "封面.png", 71L, USER).id()).isEqualTo(mediaId);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_blob WHERE resource_id = ?", Integer.class, mediaId)).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_event WHERE resource_id = ?", Integer.class, mediaId)).isEqualTo(2);
        var row = repository.get(mediaId, false);
        Path raw = directory.resolve(".afv-managed").resolve(repository.blob(row.blobId()).objectKey());
        assertThat(Arrays.copyOf(Files.readAllBytes(raw), 8)).isNotEqualTo(Arrays.copyOf(png, 8));
        String token = token(tickets.browser(mediaId, OWNER, false));
        byte[] body = http.perform(get("/api/media-content/" + mediaId).param("ticket", token))
                .andExpect(status().isOk()).andExpect(header().string("Cache-Control", "private, no-store"))
                .andReturn().getResponse().getContentAsByteArray();
        assertThat(body).isEqualTo(png);
        byte[] partial = http.perform(get("/api/media-content/" + mediaId).param("ticket", token).header("Range", "bytes=7-29"))
                .andExpect(status().isPartialContent()).andExpect(header().string("Content-Range", "bytes 7-29/" + png.length))
                .andReturn().getResponse().getContentAsByteArray();
        assertThat(partial).isEqualTo(Arrays.copyOfRange(png, 7, 30));
        http.perform(head("/api/media-content/" + mediaId).param("ticket", token)).andExpect(status().isOk())
                .andExpect(header().longValue("Content-Length", png.length));
        http.perform(get("/api/media-content/" + mediaId).param("ticket", token).header("Range", "bytes=999999-"))
                .andExpect(status().isRequestedRangeNotSatisfiable()).andExpect(header().string("Content-Range", "bytes */" + png.length));
        http.perform(get("/api/media-content/" + mediaId).param("ticket", token).header("Range", "bytes=1-2")
                        .header("If-Range", "\"old\"")).andExpect(status().isOk());
        assertThat(queries.list(71L, 1, 100, OWNER).items()).anyMatch(item -> item.id().equals(mediaId));
    }

    @Test
    void privateScopePermissionsAndExpiryAreRecheckedOutsideTheCache() throws Exception {
        var media = upload(id(), png, "private.png", null, USER);
        queries.get(media.id(), OWNER);
        assertThatThrownBy(() -> queries.get(media.id(), OTHER)).isInstanceOf(MediaException.class);
        assertThatThrownBy(() -> tickets.browser(media.id(), OTHER, false)).isInstanceOf(MediaException.class);
        var ownTicket = tickets.browser(media.id(), OWNER, false);
        var otherMedia = upload(id(), png, "other.png", null, ResourceCommandContext.user(OTHER, null));
        http.perform(get("/api/media-content/" + otherMedia.id()).param("ticket", token(ownTicket))).andExpect(status().isForbidden());
        jdbc.update("UPDATE sys_user SET status = 0 WHERE id = 71");
        http.perform(get("/api/media-content/" + media.id()).param("ticket", token(ownTicket))).andExpect(status().isForbidden());
        jdbc.update("UPDATE sys_user SET status = 1 WHERE id = 71");
        jdbc.update("UPDATE afv_media_access_ticket SET expires_at = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE resource_id = ?", media.id());
        http.perform(get("/api/media-content/" + media.id()).param("ticket", token(ownTicket))).andExpect(status().isForbidden());
    }

    @Test
    void rechecksProjectAccessAfterUploadingAndAfterTicketIssue() throws Exception {
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (71, 72, 3)");
        var actor = ResourceCommandContext.user(OTHER, null);
        var resource = upload(id(), png, "project.png", 71L, actor);
        var ticket = tickets.browser(resource.id(), OTHER, false);
        var claim = registry.admit(id(), 71L, "pending.png", resource.content(), actor);
        jdbc.update("DELETE FROM afv_project_member WHERE project_id = 71 AND user_id = 72");
        assertThatThrownBy(() -> registry.finish(claim, 100, actor)).isInstanceOf(BusinessException.class);
        assertThat(repository.get(claim.resource().id(), false).resource().state()).isEqualTo("STAGING");
        http.perform(get("/api/media-content/" + resource.id()).param("ticket", token(ticket))).andExpect(status().isForbidden());
        registry.fail(claim);
        assertThat(repository.get(claim.resource().id(), false).resource().state()).isEqualTo("FAILED");
    }

    @Test
    void replayChecksContentAndAdmissionRollsBackWithoutAStorageConfig() throws Exception {
        String key = id();
        var media = upload(key, png, "image.png", null, USER);
        assertThatThrownBy(() -> upload(key, png, "different.png", null, USER))
                .isInstanceOf(MediaException.class).hasMessageContaining("幂等键");
        assertThat(queries.get(media.id(), OWNER).state()).isEqualTo("READY");
        jdbc.update("UPDATE afv_storage_config SET is_default = FALSE");
        String failed = id();
        assertThatThrownBy(() -> upload(failed, png, "failed.png", null, USER)).isInstanceOf(MediaException.class);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_resource WHERE request_id = ?", Integer.class, failed)).isZero();
    }

    @Test
    void staleUploadCannotPublishAfterLeaseTakeover() throws Exception {
        var facts = upload(id(), png, "facts.png", null, USER).content();
        String request = id();
        var first = registry.admit(request, null, "retry.png", facts, USER);
        assertThatThrownBy(() -> registry.admit(request, null, "retry.png", facts, USER))
                .isInstanceOf(MediaException.class).hasMessageContaining("接收中");
        jdbc.update("UPDATE afv_media_resource SET lease_until = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE id = ?", first.resource().id());
        var second = registry.admit(request, null, "retry.png", facts, USER);
        assertThat(second.blobId()).isNotEqualTo(first.blobId());
        assertThat(second.epoch()).isEqualTo(first.epoch() + 1);
        assertThatThrownBy(() -> registry.finish(first, 100, USER)).isInstanceOf(MediaException.class);
        registry.fail(first);
        assertThat(repository.get(second.resource().id(), false).resource().state()).isEqualTo("STAGING");
        Path source = Files.createTempFile(directory, "retry", ".png");
        Files.write(source, png);
        long stored = blobs.write(repository.blob(second.blobId()), source);
        assertThat(registry.finish(second, stored, USER).state()).isEqualTo("READY");
        assertThat(registry.admit(request, null, "retry.png", facts, USER).replay()).isTrue();
        try (var stream = blobs.open(repository.blob(second.blobId()))) {
            ByteBuffer data = ByteBuffer.allocate(png.length);
            while (data.hasRemaining()) assertThat(stream.read(data)).isPositive();
            assertThat(data.array()).isEqualTo(png);
        }
    }

    @Test
    void concurrentSubmissionsReserveExactlyOneResource() throws Exception {
        var facts = upload(id(), png, "facts.png", null, USER).content();
        String request = id();
        CountDownLatch start = new CountDownLatch(1);
        try (var pool = Executors.newFixedThreadPool(2)) {
            var first = pool.submit(() -> { start.await(); return registry.admit(request, null, "concurrent.png", facts, USER); });
            var second = pool.submit(() -> { start.await(); return registry.admit(request, null, "concurrent.png", facts, USER); });
            start.countDown();
            int succeeded = 0;
            int conflicted = 0;
            for (var task : List.of(first, second)) {
                try { task.get(15, TimeUnit.SECONDS); succeeded++; }
                catch (ExecutionException error) {
                    assertThat(error.getCause()).isInstanceOf(MediaException.class);
                    conflicted++;
                }
            }
            assertThat(succeeded).isEqualTo(1);
            assertThat(conflicted).isEqualTo(1);
        }
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_resource WHERE request_id = ?", Integer.class, request)).isEqualTo(1);
    }

    @Test
    void expiredUploadsBecomeRetryableWithoutOverwritingLiveAttempts() throws Exception {
        var facts = upload(id(), png, "facts.png", null, USER).content();
        String key = id();
        var expired = registry.admit(key, null, "expired.png", facts, USER);
        var live = registry.admit(id(), null, "live.png", facts, USER);
        jdbc.update("UPDATE afv_media_resource SET lease_until = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE id = ?", expired.resource().id());
        assertThat(registry.expireUploads()).isEqualTo(1);
        assertThat(repository.get(expired.resource().id(), false).resource().state()).isEqualTo("FAILED");
        assertThat(repository.get(live.resource().id(), false).resource().state()).isEqualTo("STAGING");
        var retry = registry.admit(key, null, "expired.png", facts, USER);
        assertThat(retry.epoch()).isEqualTo(expired.epoch() + 1);
        assertThat(retry.blobId()).isNotEqualTo(expired.blobId());
    }

    @Test
    void eventFailureRollsBackReadyStateAndBlobBinding() throws Exception {
        var facts = upload(id(), png, "facts.png", null, USER).content();
        var claim = registry.admit(id(), null, "atomic.png", facts, USER);
        int before = jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_event WHERE resource_id = ?", Integer.class, claim.resource().id());
        doThrow(new IllegalStateException("test event failure")).when(repository).event(any(), any(), anyLong());
        assertThatThrownBy(() -> registry.finish(claim, 100, USER)).hasMessageContaining("test event failure");
        assertThat(repository.get(claim.resource().id(), false).resource().state()).isEqualTo("STAGING");
        assertThat(repository.blob(claim.blobId()).storedBytes()).isNull();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_event WHERE resource_id = ?", Integer.class, claim.resource().id())).isEqualTo(before);
    }

    @Test
    void derivesTypesFromContentAndRejectsMalformedFiles() throws Exception {
        var wrongMime = intake.upload(new ByteArrayInputStream(png), "fake.mp4", "video/mp4", id(), null, USER);
        assertThat(wrongMime.content().kind()).isEqualTo(MediaKind.IMAGE);
        var script = intake.upload(new ByteArrayInputStream("第一场\n人物：小明".getBytes(StandardCharsets.UTF_8)),
                "script.md", "text/markdown", id(), null, USER);
        assertThat(script.content().kind()).isEqualTo(MediaKind.TEXT);
        var document = intake.upload(new ByteArrayInputStream("%PDF-1.4\n1 0 obj <<>> endobj\n%%EOF".getBytes(StandardCharsets.US_ASCII)),
                "notes.pdf", "application/pdf", id(), null, USER);
        assertThat(document.content().kind()).isEqualTo(MediaKind.DOCUMENT);
        for (byte[] invalid : List.of(new byte[0], new byte[]{1, 2, 3, 4}, "%PDF-1.4\ntruncated".getBytes(StandardCharsets.US_ASCII))) {
            assertThatThrownBy(() -> intake.upload(new ByteArrayInputStream(invalid), "bad.png", "image/png", id(), null, USER))
                    .isInstanceOf(MediaException.class);
        }
        assertThatThrownBy(() -> intake.upload(new ByteArrayInputStream(new byte[]{(byte) 0xff}), "bad.txt", "text/plain", id(), null, USER))
                .isInstanceOf(MediaException.class);
        assertThatThrownBy(() -> intake.upload(new ByteArrayInputStream("{} {}".getBytes(StandardCharsets.UTF_8)), "bad.json", "application/json", id(), null, USER))
                .isInstanceOf(MediaException.class);
    }

    @Test
    void probesRealVideoAndAudioAndReadsVideoSeek() throws Exception {
        Path video = directory.resolve("probe-" + id() + ".mp4");
        Process ffmpeg = new ProcessBuilder("ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=12:d=1",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "libx264", "-c:a", "aac", "-shortest",
                video.toString()).redirectOutput(ProcessBuilder.Redirect.DISCARD).redirectError(ProcessBuilder.Redirect.DISCARD).start();
        assertThat(ffmpeg.waitFor(30, TimeUnit.SECONDS)).isTrue();
        assertThat(ffmpeg.exitValue()).isZero();
        var resource = intake.result(video, "clip.mp4", "video/mp4", id(), 71L, USER);
        assertThat(resource.content().kind()).isEqualTo(MediaKind.VIDEO);
        assertThat(resource.content().hasAudio()).isTrue();
        assertThat(resource.content().durationMs()).isBetween(900L, 1300L);
        var grant = tickets.browser(resource.id(), OWNER, false);
        byte[] bytes = http.perform(get("/api/media-content/" + resource.id()).param("ticket", token(grant))
                        .header("Range", "bytes=-300")).andExpect(status().isPartialContent())
                .andReturn().getResponse().getContentAsByteArray();
        byte[] original = Files.readAllBytes(video);
        assertThat(bytes).isEqualTo(Arrays.copyOfRange(original, original.length - 300, original.length));
        Path audio = directory.resolve("probe-" + id() + ".wav");
        Process wav = new ProcessBuilder("ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                audio.toString()).redirectOutput(ProcessBuilder.Redirect.DISCARD).redirectError(ProcessBuilder.Redirect.DISCARD).start();
        assertThat(wav.waitFor(30, TimeUnit.SECONDS)).isTrue();
        assertThat(wav.exitValue()).isZero();
        var sound = intake.result(audio, "voice.wav", "audio/wav", id(), null, USER);
        assertThat(sound.content().kind()).isEqualTo(MediaKind.AUDIO);
        assertThat(sound.content().durationMs()).isBetween(490L, 520L);
    }

    @Test
    void publicS3ContainsCiphertextWhileGatewayReturnsAuthorizedPlaintext() throws Exception {
        jdbc.update("UPDATE afv_storage_config SET is_default = (id = 72)");
        var resource = upload(id(), png, "s3.png", null, USER);
        var blob = repository.blob(repository.get(resource.id(), false).blobId());
        var client = context.getBean(S3ClientFactory.class).getClient(context.getBean(S3StorageConfigResolver.class).resolve(config(72)));
        client.putBucketPolicy(request -> request.bucket("media-test").policy("""
                {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":"*","Action":["s3:GetObject"],"Resource":["arn:aws:s3:::media-test/*"]}]}
                """));
        URI publicUri = URI.create("http://" + S3.getHost() + ":" + S3.getMappedPort(9000) + "/media-test/test/.afv-managed/" + blob.objectKey());
        var raw = HttpClient.newHttpClient().send(HttpRequest.newBuilder(publicUri).GET().build(), HttpResponse.BodyHandlers.ofByteArray());
        assertThat(raw.statusCode()).isEqualTo(200);
        assertThat(Arrays.copyOf(raw.body(), 8)).isNotEqualTo(Arrays.copyOf(png, 8));
        var ticket = tickets.browser(resource.id(), OWNER, false);
        byte[] decoded = http.perform(get("/api/media-content/" + resource.id()).param("ticket", token(ticket))
                        .header("Range", "bytes=5-45")).andExpect(status().isPartialContent())
                .andReturn().getResponse().getContentAsByteArray();
        assertThat(decoded).isEqualTo(Arrays.copyOfRange(png, 5, 46));
    }

    @Test
    void preventsStorageDeletionAndLocatorChangesButAllowsCredentialRotation() throws Exception {
        upload(id(), png, "guard.png", null, USER);
        var guard = context.getBean(MediaStorageConfigGuard.class);
        var current = config(71);
        assertThatThrownBy(() -> guard.assertDeletable(71L)).isInstanceOf(BusinessException.class);
        assertThatThrownBy(() -> guard.assertUpdatable(current, StorageConfig.builder().basePath(directory.resolve("moved").toString()).build()))
                .isInstanceOf(BusinessException.class);
        guard.assertUpdatable(current, StorageConfig.builder().accessKey("rotated").secretKey("rotated-test").build());
        flyway.validate();
    }
}
