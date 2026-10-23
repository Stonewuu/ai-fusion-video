package com.stonewu.fusion.integration;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.MediaProbe;
import com.stonewu.fusion.service.media.MediaSourceImportService;
import com.stonewu.fusion.service.media.MediaSourceQueries;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.storage.LegacyMediaRoot;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.doAnswer;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@Testcontainers(disabledWithoutDocker = false)
class MediaSourceImportIT extends MediaResourceTestContext {
    static final AtomicLong IDS = new AtomicLong(1000);
    MediaSourceImportService imports;
    MediaSourceQueries sources;

    @BeforeEach
    void prepareSources() {
        imports = context.getBean(MediaSourceImportService.class);
        sources = context.getBean(MediaSourceQueries.class);
        doAnswer(call -> config(((Number) call.getArgument(0)).longValue())).when(storageMapper).selectById(any());
        // 与本机 Web MVC 的历史公开路由固定相同根目录，不随之后的默认后端改变。
        assertThat(context.getBean(LegacyMediaRoot.class).path()).isEqualTo(directory.toAbsolutePath().normalize());
    }

    @Test
    void importsTheExactAssetFieldAndReusesTheSameResourceWithoutRefetching() throws Exception {
        String url = localImage();
        long asset = asset(url, null);
        var source = new MediaSourceReference(CanvasValueType.ASSET, asset, 0, "coverUrl");
        var imported = imports.importSource(source, USER);
        assertThat(imported.ownerType()).isEqualTo(1);
        assertThat(imported.ownerId()).isEqualTo(71);
        assertThat(imported.projectId()).isNull();
        assertThat(imported.content().kind()).isEqualTo(MediaKind.IMAGE);
        assertThat(imported.name()).endsWith(".png");
        assertThat(queries.sources(imported.id(), OWNER)).containsExactly(source);
        Files.delete(directory.resolve(url.substring("/media/".length())));
        assertThat(imports.importSource(source, USER).id()).isEqualTo(imported.id());
        String json = mapper.writeValueAsString(imported);
        assertThat(json).doesNotContain(url, "keyset", "objectKey");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_source WHERE resource_id = ?", Integer.class, imported.id())).isEqualTo(1);
        var ticket = tickets.browser(imported.id(), OWNER, false);
        byte[] body = http.perform(get("/api/media-content/" + imported.id()).param("ticket", token(ticket)))
                .andExpect(status().isOk()).andReturn().getResponse().getContentAsByteArray();
        assertThat(body).isEqualTo(png);
    }

    @Test
    void inheritsPersonalOwnershipWhenImportedByATeamPeerAndRevokesPeerAccess() throws Exception {
        long team = team();
        String url = localImage();
        long asset = asset(url, null);
        var ownerInTeam = ResourceCommandContext.user(new CanvasActor(71, team), null);
        var peer = ResourceCommandContext.user(new CanvasActor(72, team), null);
        var ref = new MediaSourceReference(CanvasValueType.ASSET, asset, 0, "coverUrl");
        var imported = imports.importSource(ref, peer);
        assertThat(imported.ownerType()).isEqualTo(1);
        assertThat(imported.ownerId()).isEqualTo(71);
        assertThat(imported.createdBy()).isEqualTo(72);
        assertThat(imports.importSource(ref, ownerInTeam).id()).isEqualTo(imported.id());
        assertThat(queries.get(imported.id(), peer.actor()).id()).isEqualTo(imported.id());
        var ticket = tickets.browser(imported.id(), peer.actor(), false);
        jdbc.update("UPDATE afv_team_member SET status = 0 WHERE team_id = ? AND user_id = 72", team);
        assertThatThrownBy(() -> queries.get(imported.id(), peer.actor())).isInstanceOf(BusinessException.class);
        assertThatThrownBy(() -> imports.importSource(ref, peer)).isInstanceOf(BusinessException.class);
        http.perform(get("/api/media-content/" + imported.id()).param("ticket", token(ticket))).andExpect(status().isForbidden());
        assertThat(queries.get(imported.id(), OWNER).id()).isEqualTo(imported.id());
    }

    @Test
    void importingAProjectSourceRequiresReadAccessAndDoesNotChangeTheBusinessVersion() throws Exception {
        long asset = asset(localImage(), 71L);
        var ref = new MediaSourceReference(CanvasValueType.ASSET, asset, 0, "coverUrl");
        assertThatThrownBy(() -> imports.importSource(ref, ResourceCommandContext.user(OTHER, null))).isInstanceOf(BusinessException.class);
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (71, 72, 3)");
        var imported = imports.importSource(ref, ResourceCommandContext.user(OTHER, null));
        assertThat(imported.projectId()).isEqualTo(71);
        assertThat(imported.ownerId()).isEqualTo(71);
        assertThat(jdbc.queryForObject("SELECT version FROM afv_asset WHERE id = ?", Integer.class, asset)).isZero();
        jdbc.update("DELETE FROM afv_project_member WHERE project_id = 71 AND user_id = 72");
        assertThatThrownBy(() -> queries.get(imported.id(), OTHER)).isInstanceOf(BusinessException.class);
    }

    @Test
    void sourceChangesDuringObjectWritingCannotPublishStaleMedia() throws Exception {
        long asset = asset(localImage(), null);
        var ref = new MediaSourceReference(CanvasValueType.ASSET, asset, 0, "coverUrl");
        AtomicBoolean changed = new AtomicBoolean();
        doAnswer(call -> {
            if (changed.compareAndSet(false, true)) jdbc.update("UPDATE afv_asset SET version = version + 1 WHERE id = ?", asset);
            return config(((Number) call.getArgument(0)).longValue());
        }).when(storageMapper).selectById(any());
        assertThatThrownBy(() -> imports.importSource(ref, USER)).isInstanceOf(MediaException.class).hasMessageContaining("来源");
        var failed = repository.source(ref, false).orElseThrow().resource();
        assertThat(failed.state()).isEqualTo("FAILED");
        assertThatThrownBy(() -> tickets.browser(failed.id(), OWNER, false)).isInstanceOf(MediaException.class);
        assertThatThrownBy(() -> imports.importSource(ref, USER)).isInstanceOf(MediaException.class);
        var next = imports.importSource(new MediaSourceReference(CanvasValueType.ASSET, asset, 1, "coverUrl"), USER);
        assertThat(next.state()).isEqualTo("READY");
        assertThat(next.id()).isNotEqualTo(failed.id());
    }

    @Test
    void sameVersionChangesAreDetectedInsteadOfReturningTheOldFile() throws Exception {
        long asset = asset(localImage(), null);
        var ref = new MediaSourceReference(CanvasValueType.ASSET, asset, 0, "coverUrl");
        imports.importSource(ref, USER);
        jdbc.update("UPDATE afv_asset SET cover_url = ? WHERE id = ?", localImage(), asset);
        assertThatThrownBy(() -> imports.importSource(ref, USER)).isInstanceOf(MediaException.class).hasMessageContaining("同一业务版本");
    }

    @Test
    void simultaneousImportsByDifferentActorsProduceOneBindingAndOneBlob() throws Exception {
        long team = team();
        long asset = asset(localImage(), null);
        var ref = new MediaSourceReference(CanvasValueType.ASSET, asset, 0, "coverUrl");
        var peer = ResourceCommandContext.user(new CanvasActor(72, team), null);
        var owner = ResourceCommandContext.user(new CanvasActor(71, team), null);
        CountDownLatch start = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(2)) {
            var first = executor.submit(() -> { start.await(); return imports.importSource(ref, owner); });
            var second = executor.submit(() -> { start.await(); return imports.importSource(ref, peer); });
            int success = 0;
            for (var future : List.of(first, second)) {
                start.countDown();
                try {
                    assertThat(future.get(20, TimeUnit.SECONDS).state()).isEqualTo("READY");
                    success++;
                } catch (ExecutionException error) {
                    assertThat(error.getCause()).isInstanceOf(MediaException.class);
                    assertThat(((MediaException) error.getCause()).getErrorCode()).isEqualTo("MEDIA_INGEST_IN_PROGRESS");
                }
            }
            assertThat(success).isGreaterThanOrEqualTo(1);
        }
        MediaResource imported = imports.importSource(ref, owner);
        assertThat(imports.importSource(ref, peer).id()).isEqualTo(imported.id());
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_blob WHERE resource_id = ?", Integer.class, imported.id())).isEqualTo(1);
    }

    @Test
    void declaredSourceModesAndRealContentMustAgree() throws Exception {
        long asset = asset(localImage(), null);
        long item = IDS.incrementAndGet();
        String url = localImage();
        jdbc.update("INSERT INTO afv_asset_item(id, asset_id, name, image_url, properties) VALUES (?, ?, '变体', ?, ?)",
                item, asset, url, mapper.writeValueAsString(mapper.createObjectNode().put("sourceGenerationMode", "video").put("sourceMediaUrl", url)));
        var video = new MediaSourceReference(CanvasValueType.ASSET_ITEM, item, 0, "properties.sourceMediaUrl");
        assertThatThrownBy(() -> imports.importSource(video, USER)).isInstanceOf(MediaException.class).hasMessageContaining("类别");
        assertThat(repository.source(video, false)).isEmpty();
        jdbc.update("UPDATE afv_asset_item SET properties = ? WHERE id = ?", "{\"sourceMediaUrl\":\"" + url + "\"}", item);
        assertThatThrownBy(() -> imports.importSource(video, USER)).isInstanceOf(MediaException.class).hasMessageContaining("sourceGenerationMode");
        jdbc.update("UPDATE afv_asset_item SET version = 1, properties = ? WHERE id = ?",
                mapper.writeValueAsString(mapper.createObjectNode().put("sourceGenerationMode", "image").put("sourceMediaUrl", url)), item);
        var image = imports.importSource(new MediaSourceReference(CanvasValueType.ASSET_ITEM, item, 1, "properties.sourceMediaUrl"), USER);
        assertThat(image.content().kind()).isEqualTo(MediaKind.IMAGE);
    }

    @Test
    void mapsStoryboardFramesVideoPosterCompositionAndAudioWithoutGuessingOtherFields() throws Exception {
        long board = IDS.incrementAndGet();
        long episode = IDS.incrementAndGet();
        long scene = IDS.incrementAndGet();
        long shot = IDS.incrementAndGet();
        String url = localImage();
        jdbc.update("INSERT INTO afv_storyboard(id, project_id, title) VALUES (?, 71, '分镜')", board);
        jdbc.update("INSERT INTO afv_storyboard_episode(id, storyboard_id, episode_number, title, composed_video_url, bgm_audio_url, bgm_sfx_audio_url) VALUES (?, ?, 1, '第一集', '/media/result.mp4', '/media/music.mp3', '/media/sfx.wav')", episode, board);
        jdbc.update("INSERT INTO afv_storyboard_scene(id, storyboard_id, episode_id, scene_heading) VALUES (?, ?, ?, '场景')", scene, board, episode);
        jdbc.update("""
                INSERT INTO afv_storyboard_item(id, storyboard_id, storyboard_episode_id, storyboard_scene_id,
                    first_frame_image_url, last_frame_image_url, generated_video_url, custom_data)
                VALUES (?, ?, ?, ?, ?, ?, '/media/shot.mp4', ?)
                """, shot, board, episode, scene, url, url, "{\"videoPosterUrl\":\"" + url + "\"}");
        for (String field : List.of("firstFrameImageUrl", "lastFrameImageUrl", "customData.videoPosterUrl")) {
            var resource = imports.importSource(new MediaSourceReference(CanvasValueType.SHOT, shot, 0, field), USER);
            assertThat(resource.projectId()).isEqualTo(71);
            assertThat(resource.content().kind()).isEqualTo(MediaKind.IMAGE);
        }
        assertThat(sources.resolve(new MediaSourceReference(CanvasValueType.SHOT, shot, 0, "generatedVideoUrl"), USER).kind()).isEqualTo(MediaKind.VIDEO);
        assertThat(sources.resolve(new MediaSourceReference(CanvasValueType.STORYBOARD_EPISODE, episode, 0, "composedVideoUrl"), USER).kind()).isEqualTo(MediaKind.VIDEO);
        for (String field : List.of("bgmAudioUrl", "bgmSfxAudioUrl")) {
            assertThat(sources.resolve(new MediaSourceReference(CanvasValueType.STORYBOARD_EPISODE, episode, 0, field), USER).kind()).isEqualTo(MediaKind.AUDIO);
        }
        assertThatThrownBy(() -> sources.resolve(new MediaSourceReference(CanvasValueType.SHOT, shot, 0, "content"), USER))
                .isInstanceOf(MediaException.class);
    }

    @Test
    void failedSourceCanBeRetriedByAnotherAuthorizedActorWithItsOwnAttempt() throws Exception {
        long team = team();
        long asset = asset(localImage(), null);
        var ref = new MediaSourceReference(CanvasValueType.ASSET, asset, 0, "coverUrl");
        var observed = sources.resolve(ref, USER);
        Path file = directory.resolve(observed.uri().substring("/media/".length()));
        var facts = context.getBean(MediaProbe.class).inspect(file, "application/octet-stream");
        var first = registry.admitSource(observed, facts, USER);
        registry.fail(first);
        var peer = ResourceCommandContext.user(new CanvasActor(72, team), null);
        var result = imports.importSource(ref, peer);
        assertThat(result.id()).isEqualTo(first.resource().id());
        assertThat(result.ownerId()).isEqualTo(71);
        assertThat(result.createdBy()).isEqualTo(71);
        assertThat(jdbc.queryForObject("SELECT actor_user_id FROM afv_media_event WHERE resource_id = ? AND event_type = 'media.ready'", Long.class, result.id())).isEqualTo(72);
    }

    private String localImage() throws Exception {
        String name = id() + ".png";
        Files.write(directory.resolve(name), png);
        return "/media/" + name;
    }

    @Test
    void agentCancellationFencesBothImportCompletionAndReadyReplay() throws Exception {
        long asset = asset(localImage(), 71L);
        var ref = new MediaSourceReference(CanvasValueType.ASSET, asset, 0, "coverUrl");
        String run = id();
        jdbc.update("""
                INSERT INTO afv_agent_run(run_id, conversation_id, user_id, project_id, kernel_fingerprint,
                    agent_definition_snapshot_json, agent_state_session_id, status, owner_instance_id, owner_epoch,
                    lease_until, deadline_at, started_at)
                VALUES (?, ?, 71, 71, 'media-test', '{}', ?, 'RUNNING', 'media-owner', 1,
                    TIMESTAMPADD(MINUTE, 10, UTC_TIMESTAMP(6)), TIMESTAMPADD(HOUR, 1, UTC_TIMESTAMP(6)), UTC_TIMESTAMP(6))
                """, run, run, run);
        var agent = new ResourceCommandContext(OWNER, "agent", null, run, "source-import", "media-owner", 1L);
        var result = imports.importSource(ref, agent);
        jdbc.update("UPDATE afv_agent_run SET cancel_requested_at = UTC_TIMESTAMP(6) WHERE run_id = ?", run);
        assertThatThrownBy(() -> imports.importSource(ref, agent)).isInstanceOf(BusinessException.class);
        assertThat(imports.importSource(ref, USER).id()).isEqualTo(result.id());

        jdbc.update("UPDATE afv_agent_run SET cancel_requested_at = NULL WHERE run_id = ?", run);
        var second = new MediaSourceReference(CanvasValueType.ASSET, asset(localImage(), 71L), 0, "coverUrl");
        AtomicBoolean cancelled = new AtomicBoolean();
        doAnswer(call -> {
            if (cancelled.compareAndSet(false, true)) jdbc.update("UPDATE afv_agent_run SET cancel_requested_at = UTC_TIMESTAMP(6) WHERE run_id = ?", run);
            return config(((Number) call.getArgument(0)).longValue());
        }).when(storageMapper).selectById(any());
        assertThatThrownBy(() -> imports.importSource(second, agent)).isInstanceOf(BusinessException.class);
        assertThat(repository.source(second, false).orElseThrow().resource().state()).isEqualTo("FAILED");
    }
    private long asset(String url, Long projectId) {
        long id = IDS.incrementAndGet();
        jdbc.update("INSERT INTO afv_asset(id, name, type, project_id, user_id, owner_type, owner_id, cover_url) VALUES (?, '来源', 'image', ?, 71, 1, 71, ?)", id, projectId, url);
        return id;
    }
    private long team() {
        long id = IDS.incrementAndGet();
        jdbc.update("INSERT INTO afv_team(id, name, owner_user_id) VALUES (?, '源素材团队', 71)", id);
        jdbc.update("INSERT INTO afv_team_member(team_id, user_id, role) VALUES (?, 71, 1), (?, 72, 3)", id, id);
        return id;
    }
}
