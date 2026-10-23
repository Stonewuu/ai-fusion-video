package com.stonewu.fusion.integration;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.media.MediaSourceImportService;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.media.preview.*;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.dao.DataAccessResourceFailureException;
import org.testcontainers.junit.jupiter.Testcontainers;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicBoolean;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.reset;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@Testcontainers(disabledWithoutDocker = false)
class MediaPreviewIT extends MediaResourceTestContext {
    private static final AtomicLong IDS = new AtomicLong(69000);
    private static final MediaPreviewData.Request ENSURE = new MediaPreviewData.Request(null);
    private MediaPreviewService previews;
    private MediaPreviewQueries views;
    private MediaPreviewJobs jobs;
    private MediaPreviewWorker worker;
    private GenerationOutbox outbox;

    @BeforeEach void preparePreviews() {
        previews = context.getBean(MediaPreviewService.class);
        views = context.getBean(MediaPreviewQueries.class);
        jobs = context.getBean(MediaPreviewJobs.class);
        worker = context.getBean(MediaPreviewWorker.class);
        outbox = context.getBean(GenerationOutbox.class);
        jdbc.update("UPDATE afv_media_job SET status = 'CANCELLED' WHERE kind = 'MEDIA_PREVIEW'");
        doAnswer(call -> config(((Number) call.getArgument(0)).longValue())).when(storageMapper).selectById(any());
    }

    @Test void imagePreviewIsDurableEncryptedDeduplicatedAndHiddenFromTheOrdinaryLibrary() throws Exception {
        var source = upload(id(), png, "原图.png", null, USER);
        var queued = previews.ensure(source.id(), ENSURE, OWNER);
        assertThat(queued.status()).isEqualTo("QUEUED");
        assertThat(previews.ensure(source.id(), ENSURE, OWNER).jobId()).isEqualTo(queued.jobId());
        assertThat(worker.runOne()).isFalse();
        outbox.dispatch(); worker.runOne();
        var ready = views.get(source.id(), OWNER);
        assertThat(ready.status()).isEqualTo("SUCCEEDED");
        assertThat(ready.width()).isEqualTo(16);
        assertThat(ready.height()).isEqualTo(10);
        assertThat(ready.previewId()).isNotEqualTo(source.id());
        var child = queries.get(ready.previewId(), OWNER);
        assertThat(child.content().kind()).isEqualTo(MediaKind.IMAGE);
        assertThat(child.ownerId()).isEqualTo(source.ownerId());
        assertThat(queries.list(null, 1, 100, OWNER).items()).noneMatch(item -> item.id().equals(child.id()));
        var row = repository.get(child.id(), false);
        Path ciphertext = directory.resolve(".afv-managed").resolve(repository.blob(row.blobId()).objectKey());
        assertThat(Arrays.copyOf(Files.readAllBytes(ciphertext), 8)).isNotEqualTo(Arrays.copyOf(png, 8));
        var ticket = tickets.browser(child.id(), OWNER, false);
        byte[] content = http.perform(get("/api/media-content/" + child.id()).param("ticket", token(ticket))).andExpect(status().isOk())
                .andReturn().getResponse().getContentAsByteArray();
        assertThat(Arrays.copyOf(content, 8)).isEqualTo(Arrays.copyOf(png, 8));
        assertThat(previews.ensure(source.id(), ENSURE, OWNER).previewId()).isEqualTo(child.id());
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_reference WHERE media_id = ? AND anchor_kind = 'PREVIEW_SOURCE' AND anchor_id = ?", Integer.class, source.id(), queued.jobId())).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_reference WHERE media_id = ? AND anchor_kind = 'PREVIEW_OUTPUT' AND anchor_id = ?", Integer.class, child.id(), queued.jobId())).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_reference WHERE media_id = ? AND anchor_kind = 'ACCESS_TICKET'", Integer.class, child.id())).isGreaterThan(0);
        jdbc.update("UPDATE afv_media_access_ticket SET expires_at = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE resource_id = ?", child.id());
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_reference WHERE media_id = ? AND anchor_kind = 'ACCESS_TICKET'", Integer.class, child.id())).isZero();
        assertThatThrownBy(() -> previews.ensure(child.id(), ENSURE, OWNER)).isInstanceOf(MediaException.class);
    }

    @Test void parallelReadersShareOneAcceptedPreviewJob() throws Exception {
        var source = upload(id(), png, "并行原图.png", 71L, USER);
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (71, 72, 3)");
        try (var executor = Executors.newFixedThreadPool(2)) {
            var a = executor.submit(() -> previews.ensure(source.id(), ENSURE, OWNER));
            var b = executor.submit(() -> previews.ensure(source.id(), ENSURE, OTHER));
            assertThat(a.get(20, TimeUnit.SECONDS).jobId()).isEqualTo(b.get(20, TimeUnit.SECONDS).jobId());
        }
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_preview_job WHERE source_resource_id = ?", Integer.class, source.id())).isEqualTo(1);
        outbox.dispatch(); worker.runOne();
        assertThat(views.get(source.id(), OTHER).status()).isEqualTo("SUCCEEDED");
    }

    @Test void readOnlyProjectMemberCanGenerateAPreviewButLosesItsTicketAfterRevocation() throws Exception {
        var source = upload(id(), png, "项目原图.png", 71L, USER);
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (71, 72, 3)");
        previews.ensure(source.id(), ENSURE, OTHER);
        outbox.dispatch(); worker.runOne();
        var ready = views.get(source.id(), OTHER);
        var child = queries.get(ready.previewId(), OTHER);
        assertThat(child.ownerId()).isEqualTo(71);
        assertThat(child.createdBy()).isEqualTo(72);
        var ticket = tickets.browser(child.id(), OTHER, false);
        jdbc.update("DELETE FROM afv_project_member WHERE project_id = 71 AND user_id = 72");
        assertThatThrownBy(() -> queries.get(child.id(), OTHER)).isInstanceOf(BusinessException.class);
        http.perform(get("/api/media-content/" + child.id()).param("ticket", token(ticket))).andExpect(status().isForbidden());
        assertThat(views.get(source.id(), OWNER).previewId()).isEqualTo(child.id());
    }

    @Test void independentAssetPreviewAlwaysChecksTheOriginalEvenIfChildOwnershipWouldGrantAccess() throws Exception {
        long team = IDS.incrementAndGet(), asset = IDS.incrementAndGet();
        jdbc.update("INSERT INTO afv_team(id,name,owner_user_id) VALUES (?, '预览团队', 71)", team);
        jdbc.update("INSERT INTO afv_team_member(team_id,user_id,role) VALUES (?,71,1),(?,72,3)", team, team);
        String filename = id() + ".png";
        Files.write(directory.resolve(filename), png);
        jdbc.update("INSERT INTO afv_asset(id,name,type,user_id,owner_type,owner_id,cover_url) VALUES (?, '团队可读资产', 'image',71,1,71,?)", asset, "/media/" + filename);
        var peer = new CanvasActor(72, team);
        var source = context.getBean(MediaSourceImportService.class).importSource(new MediaSourceReference(CanvasValueType.ASSET, asset, 0, "coverUrl"),
                ResourceCommandContext.user(peer, null));
        previews.ensure(source.id(), ENSURE, peer);
        outbox.dispatch(); worker.runOne();
        String child = views.get(source.id(), peer).previewId();
        assertThat(queries.get(child, peer).ownerId()).isEqualTo(71);
        var ticket = tickets.browser(child, peer, false);
        jdbc.update("UPDATE afv_media_resource SET owner_id = 72 WHERE id = ?", child);
        jdbc.update("UPDATE afv_team_member SET status = 0 WHERE team_id = ? AND user_id = 71", team);
        assertThatThrownBy(() -> queries.get(child, peer)).isInstanceOf(BusinessException.class);
        http.perform(get("/api/media-content/" + child).param("ticket", token(ticket))).andExpect(status().isForbidden());
    }

    @Test void finishedFileIsReusedAfterWorkerRestartWithoutRunningTheDecoderAgain() throws Exception {
        var source = upload(id(), png, "恢复原图.png", null, USER);
        var queued = previews.ensure(source.id(), ENSURE, OWNER);
        outbox.dispatch();
        var original = jobs.claim("before-restart").orElseThrow();
        Path output = directory.resolve(id() + ".png"); Files.write(output, png);
        String child = context.getBean(MediaPreviewIntake.class).store(original, output).id();
        jdbc.update("UPDATE afv_media_job SET lease_until = TIMESTAMPADD(SECOND,-1,UTC_TIMESTAMP(6)) WHERE id = ?", queued.jobId());
        Object previous = ReflectionTestUtils.getField(worker, "ffmpeg");
        ReflectionTestUtils.setField(worker, "ffmpeg", "not-a-real-decoder");
        try { assertThat(worker.runOne()).isTrue(); }
        finally { ReflectionTestUtils.setField(worker, "ffmpeg", previous); }
        assertThat(views.get(source.id(), OWNER).previewId()).isEqualTo(child);
        assertThatThrownBy(() -> jobs.complete(original.claim(), child)).isInstanceOf(MediaException.class);
    }

    @Test void disabledOriginalReaderStopsItsJobAndAnotherAuthorizedReaderCanExplicitlyRetry() throws Exception {
        var source = upload(id(), png, "接管原图.png", 71L, USER);
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (71,72,3)");
        var first = previews.ensure(source.id(), ENSURE, OWNER);
        outbox.dispatch(); jdbc.update("UPDATE sys_user SET status = 0 WHERE id = 71"); worker.runOne();
        var stopped = views.get(source.id(), OTHER);
        assertThat(stopped.status()).isEqualTo("CANCELLED");
        var request = new MediaPreviewData.Request(first.jobId());
        var retry = previews.ensure(source.id(), request, OTHER);
        assertThat(retry.jobId()).isNotEqualTo(first.jobId());
        assertThat(previews.ensure(source.id(), request, OTHER).jobId()).isEqualTo(retry.jobId());
        outbox.dispatch(); worker.runOne();
        assertThat(views.get(source.id(), OTHER).status()).isEqualTo("SUCCEEDED");
        assertThat(previews.ensure(source.id(), request, OTHER).jobId()).isEqualTo(retry.jobId());
    }

    @Test void interruptedStagingReceiverIsReclaimedWithTheNewJobEpochWithoutWaitingForUploadExpiry() throws Exception {
        var source = upload(id(), png, "写入中断原图.png", null, USER);
        var job = previews.ensure(source.id(), ENSURE, OWNER); outbox.dispatch();
        AtomicBoolean crashed = new AtomicBoolean();
        doAnswer(call -> {
            String blobId = call.getArgument(0);
            var children = jdbc.queryForList("SELECT r.blob_id FROM afv_media_preview_job p JOIN afv_media_resource r ON r.id = p.output_resource_id WHERE p.job_id = ?", String.class, job.jobId());
            if (children.contains(blobId)) { crashed.set(true); throw new DataAccessResourceFailureException("模拟写入时数据库中断"); }
            return call.callRealMethod();
        }).when(repository).blob(anyString());
        doAnswer(call -> {
            if (crashed.get() && (boolean) call.getArgument(1)) throw new DataAccessResourceFailureException("模拟清理也无法提交");
            return call.callRealMethod();
        }).when(repository).get(anyString(), anyBoolean());
        worker.runOne();
        reset(repository);
        assertThat(crashed).isTrue();
        String child = jdbc.queryForObject("SELECT output_resource_id FROM afv_media_preview_job WHERE job_id = ?", String.class, job.jobId());
        var before = repository.get(child, false);
        assertThat(before.resource().state()).isEqualTo("STAGING");
        jdbc.update("UPDATE afv_media_job SET lease_until = TIMESTAMPADD(SECOND,-1,UTC_TIMESTAMP(6)) WHERE id = ?", job.jobId());
        worker.runOne();
        assertThat(views.get(source.id(), OWNER).status()).isEqualTo("SUCCEEDED");
        assertThat(views.get(source.id(), OWNER).previewId()).isEqualTo(child);
        assertThat(repository.get(child, false).epoch()).isGreaterThan(before.epoch());
        assertThat(jdbc.queryForObject("SELECT state FROM afv_media_blob WHERE id = ?", String.class, before.blobId())).isEqualTo("ORPHAN");
    }

    @Test void changesToTheOriginalVersionInvalidateExistingPreviewTickets() throws Exception {
        var source = upload(id(), png, "版本原图.png", null, USER);
        previews.ensure(source.id(), ENSURE, OWNER); outbox.dispatch(); worker.runOne();
        String child = views.get(source.id(), OWNER).previewId();
        var ticket = tickets.browser(child, OWNER, false);
        jdbc.update("UPDATE afv_media_resource SET version = version + 1 WHERE id = ?", source.id());
        assertThatThrownBy(() -> queries.get(child, OWNER)).isInstanceOf(MediaException.class).hasMessageContaining("失效");
        http.perform(get("/api/media-content/" + child).param("ticket", token(ticket))).andExpect(status().isGone());
        assertThat(views.get(source.id(), OWNER).status()).isEqualTo("ABSENT");
    }

    @Test void alteredFrozenInputCannotPublishAnotherPrivateFileAsThisSourcesPreview() throws Exception {
        var a = upload(id(), png, "原图甲.png", null, USER);
        var b = upload(id(), png, "原图乙.png", null, USER);
        var job = previews.ensure(a.id(), ENSURE, OWNER);
        jdbc.update("UPDATE afv_media_job SET input_json = JSON_SET(input_json, '$.source.id', ?) WHERE id = ?", b.id(), job.jobId());
        outbox.dispatch(); worker.runOne();
        var result = views.get(a.id(), OWNER);
        assertThat(result.status()).isEqualTo("FAILED");
        assertThat(result.previewId()).isNull();
        assertThat(jdbc.queryForObject("SELECT output_resource_id FROM afv_media_preview_job WHERE job_id = ?", String.class, job.jobId())).isNull();
    }

    @Test void decoderFailureIsExplicitAndDoesNotAutoRetryOrReturnTheOriginalFile() throws Exception {
        var source = upload(id(), png, "处理失败原图.png", null, USER);
        var job = previews.ensure(source.id(), ENSURE, OWNER);
        outbox.dispatch();
        Object previous = ReflectionTestUtils.getField(worker, "ffmpeg");
        ReflectionTestUtils.setField(worker, "ffmpeg", "not-a-real-decoder");
        try { worker.runOne(); } finally { ReflectionTestUtils.setField(worker, "ffmpeg", previous); }
        var failed = previews.ensure(source.id(), ENSURE, OWNER);
        assertThat(failed.status()).isEqualTo("FAILED");
        assertThat(failed.previewId()).isNull();
        assertThat(failed.jobId()).isEqualTo(job.jobId());
        assertThat(worker.runOne()).isFalse();
        previews.ensure(source.id(), new MediaPreviewData.Request(job.jobId()), OWNER);
        outbox.dispatch(); worker.runOne();
        assertThat(views.get(source.id(), OWNER).status()).isEqualTo("SUCCEEDED");
    }

    @Test void videoPreviewRecordsItsSamplingPositionAndProducesAnImageWithBoundedDimensions() throws Exception {
        Path video = directory.resolve(id() + ".mp4");
        var command = new ArrayList<>(List.of("ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", "color=c=blue:s=960x640:r=10:d=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", video.toString()));
        Process process = new ProcessBuilder(command).redirectErrorStream(true).redirectOutput(directory.resolve("preview-video.log").toFile()).start();
        assertThat(process.waitFor(20, TimeUnit.SECONDS)).isTrue(); assertThat(process.exitValue()).isZero();
        var source = intake.result(video, "原视频.mp4", "video/mp4", id(), null, USER);
        previews.ensure(source.id(), ENSURE, OWNER); outbox.dispatch(); worker.runOne();
        var result = views.get(source.id(), OWNER);
        assertThat(result.status()).isEqualTo("SUCCEEDED");
        assertThat(result.sampleAtMs()).isEqualTo(source.content().durationMs() / 10);
        assertThat(result.width()).isEqualTo(640);
        assertThat(result.height()).isLessThanOrEqualTo(480);
        assertThat(queries.get(result.previewId(), OWNER).content().mimeType()).isEqualTo("image/png");
    }
}
