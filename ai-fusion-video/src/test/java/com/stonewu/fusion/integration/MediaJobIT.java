package com.stonewu.fusion.integration;

import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;

import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.config.EpisodeBgmProperties;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.sonilo.SoniloAudioClient;
import com.stonewu.fusion.service.ai.sonilo.SoniloAudioTrack;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.ProviderMediaFetcher;
import com.stonewu.fusion.service.media.job.*;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.storyboard.EpisodeBgmService;
import com.stonewu.fusion.service.storyboard.VideoComposeService;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.cache.CacheManager;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;
import org.testcontainers.junit.jupiter.Testcontainers;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** 以真实数据库、加密文件和 FFmpeg 验证旧后台合成路径的替换。 */
@Testcontainers
class MediaJobIT extends MediaResourceTestContext {
    private static final AtomicLong IDS = new AtomicLong(22000);
    static MediaJobRepository jobs;
    static VideoComposeService compose;
    static MediaJobWorker worker;
    static MediaJobCompletion completion;
    static ApiConfigService configs;
    static SoniloAudioClient sonilo;
    static ProviderMediaFetcher fetcher;
    static CanvasJson json;
    static EpisodeAudioJobs audio;
    static EpisodeAudioReviewService reviews;
    static String silentMedia;
    static String audioMedia;
    static Path silentFile;
    static Path musicFile;

    @BeforeAll static void services() throws Exception {
        configs = mock(ApiConfigService.class);
        sonilo = mock(SoniloAudioClient.class);
        fetcher = mock(ProviderMediaFetcher.class);
        context.getBeanFactory().registerSingleton("apiConfigService", configs);
        context.getBeanFactory().registerSingleton("soniloAudioClient", sonilo);
        context.getBeanFactory().registerSingleton("providerMediaFetcher", fetcher);
        context.register(EpisodeBgmProperties.class, EpisodeBgmService.class, VideoComposeService.class,
                MediaJobOutputs.class, MediaJobQueries.class, MediaJobCompletion.class,
                ComposeVideoProcessor.class, ComposeBgmProcessor.class, EpisodeAudioJobs.class, MediaJobWorker.class,
                EpisodeAudioReviewService.class, EpisodeComposeGuard.class);
        jobs = context.getBean(MediaJobRepository.class);
        compose = context.getBean(VideoComposeService.class);
        worker = context.getBean(MediaJobWorker.class);
        completion = context.getBean(MediaJobCompletion.class);
        json = context.getBean(CanvasJson.class);
        audio = context.getBean(EpisodeAudioJobs.class);
        reviews = context.getBean(EpisodeAudioReviewService.class);
        silentFile = directory.resolve("silent.mp4");
        Path withAudio = directory.resolve("with-audio.mp4");
        musicFile = directory.resolve("music.m4a");
        ffmpeg("-f", "lavfi", "-i", "color=c=blue:s=96x64:r=12:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", silentFile.toString());
        ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x96:r=25:d=1", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", withAudio.toString());
        ffmpeg("-f", "lavfi", "-i", "sine=frequency=880:duration=1", "-c:a", "aac", musicFile.toString());
        silentMedia = intake.result(silentFile, "silent.mp4", "video/mp4", id(), 71L, USER).id();
        audioMedia = intake.result(withAudio, "with-audio.mp4", "video/mp4", id(), 71L, USER).id();
    }
    @BeforeEach void cleanJobs() {
        jdbc.update("UPDATE afv_media_job SET status = 'FAILED' WHERE status IN ('QUEUED','RUNNING','WAITING')");
        jdbc.update("UPDATE afv_episode_audio_job SET permit_held = FALSE");
        context.getBean(EpisodeBgmProperties.class).setSfxEnabled(false);
        context.getBean(CacheManager.class).getCache("episodeBgmConfig").clear();
        reset(configs, sonilo, fetcher);
        when(configs.getListByPlatform("sonilo")).thenReturn(List.of());
    }

    @Test void admissionIsAtomicVersionedIdempotentAndNeverSkipsAMissingShot() {
        var fixture = episode(true);
        String request = id();
        String job = compose.submitCompose(fixture.episode(), 0, request, USER, true);
        assertThat(compose.submitCompose(fixture.episode(), 0, request, USER, true)).isEqualTo(job);
        assertThat(jobs.get(job).targetVersion()).isEqualTo(1);
        assertThat(jobs.get(job).input().path("clips")).hasSize(2);
        assertThatThrownBy(() -> compose.submitCompose(fixture.episode(), 0, id(), USER, true)).isInstanceOf(CanvasException.class);
        assertThatThrownBy(() -> compose.submitCompose(fixture.episode(), 1, id(), USER, true)).isInstanceOf(MediaException.class);
        assertThatThrownBy(() -> compose.submitCompose(fixture.episode(), 0, request, ResourceCommandContext.user(OTHER, null), true)).isInstanceOf(CanvasException.class);
        var missing = episode(false);
        jdbc.update("UPDATE afv_storyboard_item SET video_url = NULL WHERE id = ?", missing.shot());
        assertThatThrownBy(() -> compose.submitCompose(missing.episode(), 0, id(), USER, true)).hasMessageContaining("缺少视频");
        assertThat(jdbc.queryForObject("SELECT version FROM afv_storyboard_episode WHERE id = ?", Integer.class, missing.episode())).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_job WHERE target_id = ?", Integer.class, missing.episode())).isZero();
    }

    @Test void actualMixedClipsKeepSoundAndWriteOneAuditedPrivateResult() {
        var fixture = episode(true);
        String id = compose.submitCompose(fixture.episode(), 0, id(), USER, true);
        assertThat(worker.runOne()).isTrue();
        var job = jobs.get(id);
        assertThat(job.status()).as(job.error()).isEqualTo("SUCCEEDED");
        String reference = job.checkpoint().path("video").asText();
        var media = queries.get(MediaReference.id(reference), OWNER);
        assertThat(media.content().hasAudio()).isTrue();
        assertThat(media.content().durationMs()).isBetween(1900L, 2400L);
        assertThat(media.content().width()).isEqualTo(96);
        assertThat(jdbc.queryForObject("SELECT composed_video_url FROM afv_storyboard_episode WHERE id = ?", String.class, fixture.episode())).isEqualTo(reference);
        assertThat(jdbc.queryForObject("SELECT version FROM afv_storyboard_episode WHERE id = ?", Integer.class, fixture.episode())).isEqualTo(2);
        assertThat(jdbc.queryForObject("SELECT media_id FROM afv_media_binding WHERE resource_type = 'storyboardEpisode' AND resource_id = ? AND field_name = 'composedVideoUrl'",
                String.class, fixture.episode())).isEqualTo(media.id());
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_event WHERE resource_type = 'storyboardEpisode' AND resource_id = ?", Integer.class, fixture.episode())).isEqualTo(2);
        assertThat(worker.runOne()).isFalse();
    }

    @Test void expiredWorkerCannotWriteAndRestartUsesCommittedCheckpoint() throws Exception {
        var fixture = episode(false);
        String id = compose.submitCompose(fixture.episode(), 0, id(), USER, true);
        var claim = jobs.claim("dead-instance").orElseThrow().claim();
        Path work = Files.createTempDirectory(directory, "recover-");
        var prepared = context.getBean(ComposeVideoProcessor.class).prepare(claim, work);
        String base = prepared.checkpoint().path("baseVideo").asText();
        expire(id);
        assertThat(worker.runOne()).isTrue();
        assertThat(jobs.get(id).status()).isEqualTo("SUCCEEDED");
        assertThat(jobs.get(id).checkpoint().path("video").asText()).isEqualTo(base);
        assertThat(jobs.get(id).epoch()).isEqualTo(2);
        assertThatThrownBy(() -> completion.complete(claim)).isInstanceOf(MediaException.class);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_resource WHERE request_id = ?", Integer.class, "job:" + id + ":base")).isEqualTo(1);
    }

    @Test void editingShotDuringComposePreservesNewContentAndKeepsConflictingOutput() throws Exception {
        var fixture = episode(false);
        String id = compose.submitCompose(fixture.episode(), 0, id(), USER, true);
        var claim = jobs.claim("editing-instance").orElseThrow().claim();
        context.getBean(ComposeVideoProcessor.class).prepare(claim, Files.createTempDirectory(directory, "conflict-"));
        context.getBean(ResourceCommandService.class).patch(new ResourcePatchCommand(id(), CanvasValueType.SHOT, fixture.shot(), 0,
                json.object(Map.of("content", "用户正在编辑的新内容"))), USER);
        expire(id);
        worker.runOne();
        assertThat(jobs.get(id).status()).isEqualTo("CONFLICT");
        assertThat(jobs.get(id).checkpoint().path("video").asText()).startsWith("media://");
        assertThat(jdbc.queryForObject("SELECT content FROM afv_storyboard_item WHERE id = ?", String.class, fixture.shot())).isEqualTo("用户正在编辑的新内容");
        assertThat(jdbc.queryForObject("SELECT composed_video_url FROM afv_storyboard_episode WHERE id = ?", String.class, fixture.episode())).isNull();
        assertThat(jdbc.queryForObject("SELECT compose_status FROM afv_storyboard_episode WHERE id = ?", Integer.class, fixture.episode())).isEqualTo(3);
    }

    @Test void cancelledQueuedJobDoesNotStartFfmpegOrRegisterOutputs() {
        var fixture = episode(false);
        String id = compose.submitCompose(fixture.episode(), 0, id(), USER, true);
        assertThat(context.getBean(MediaJobQueries.class).cancel(id, OWNER)).isTrue();
        worker.runOne();
        assertThat(jobs.get(id).status()).isEqualTo("CANCELLED");
        assertThat(jobs.get(id).checkpoint()).isEmpty();
        assertThat(jdbc.queryForObject("SELECT compose_status FROM afv_storyboard_episode WHERE id = ?", Integer.class, fixture.episode())).isEqualTo(3);
        verifyNoInteractions(sonilo, fetcher);
    }

    @Test void unacknowledgedPaidSubmissionRequiresReviewAndIsNeverAutomaticallyRepeated() throws Exception {
        enableBgm();
        var fixture = episode(false);
        String id = compose.submitCompose(fixture.episode(), 0, id(), USER, true);
        var claim = jobs.claim("crashed-after-post").orElseThrow().claim();
        context.getBean(ComposeVideoProcessor.class).prepare(claim, Files.createTempDirectory(directory, "unknown-"));
        jobs.prepareSubmission(claim, "music");
        expire(id);
        worker.runOne();
        assertThat(jobs.get(id).status()).as(jobs.get(id).error()).isEqualTo("NEEDS_REVIEW");
        assertThat(jobs.get(id).checkpoint().path("baseVideo").asText()).startsWith("media://");
        verifyNoInteractions(sonilo, fetcher);
    }

    @Test void confirmedProviderReceiptResumesWithoutSubmittingAgainAndBindsAudio() throws Exception {
        var config = enableBgm();
        var fixture = episode(false);
        String id = compose.submitCompose(fixture.episode(), 0, id(), USER, true);
        var claim = jobs.claim("crashed-after-receipt").orElseThrow().claim();
        context.getBean(ComposeVideoProcessor.class).prepare(claim, Files.createTempDirectory(directory, "receipt-"));
        var intent = jobs.prepareSubmission(claim, "music");
        jobs.acknowledge(intent.attempt(), "known-task");
        var result = JSONUtil.parseObj("{\"status\":\"succeeded\"}");
        when(sonilo.tryQueryTask(config, "known-task")).thenReturn(result);
        when(sonilo.extractPrimaryTrack(result, "known-task")).thenReturn(new SoniloAudioTrack("https://audio.example.test/result"));
        doAnswer(call -> { Files.copy(musicFile, call.getArgument(2, Path.class), StandardCopyOption.REPLACE_EXISTING); return null; })
                .when(fetcher).copyPublic(anyString(), any(), any());
        expire(id);
        worker.runOne();
        assertThat(jobs.get(id).status()).as(jobs.get(id).error()).isEqualTo("SUCCEEDED");
        assertThat(jdbc.queryForObject("SELECT bgm_audio_url FROM afv_storyboard_episode WHERE id = ?", String.class, fixture.episode())).startsWith("media://");
        verify(sonilo, never()).submitMusicTask(any(), any(), any());
        verify(sonilo).tryQueryTask(config, "known-task");
    }

    @Test void inFlightCancellationPollsOriginalUntilEndedAndDoesNotPublishOrStartSfx() {
        var config = enableBgm();
        context.getBean(EpisodeBgmProperties.class).setSfxEnabled(true);
        var fixture = episode(false);
        String id = compose.submitCompose(fixture.episode(), 0, id(), USER, true);
        when(sonilo.submitMusicTask(eq(config), any(), any())).thenReturn("cancel-music");
        when(sonilo.tryQueryTask(config, "cancel-music")).thenReturn(JSONUtil.parseObj("{\"status\":\"processing\"}"));
        worker.runOne();
        assertThat(jobs.get(id).status()).isEqualTo("WAITING");
        assertThat(audio.reserved(id)).isTrue();
        context.getBean(MediaJobQueries.class).cancel(id, OWNER);
        worker.runOne();
        assertThat(jobs.get(id).status()).isEqualTo("WAITING");
        assertThat(audio.reserved(id)).isTrue();
        assertThat(jdbc.queryForObject("SELECT compose_status FROM afv_storyboard_episode WHERE id = ?", Integer.class, fixture.episode())).isEqualTo(1);
        when(sonilo.tryQueryTask(config, "cancel-music")).thenReturn(JSONUtil.parseObj("{\"status\":\"succeeded\"}"));
        due(id);
        worker.runOne();
        assertThat(jobs.get(id).status()).isEqualTo("CANCELLED");
        assertThat(audio.reserved(id)).isFalse();
        assertThat(jobs.get(id).checkpoint().has("music")).isFalse();
        assertThat(jobs.get(id).checkpoint().has("sfx")).isFalse();
        assertThat(jobs.get(id).checkpoint().has("video")).isFalse();
        verify(sonilo, times(1)).submitMusicTask(eq(config), any(), any());
        verify(sonilo, never()).submitSfxTask(any(), any(), any());
        verifyNoInteractions(fetcher);
    }

    @Test void unknownCancellationKeepsCapacityAndRequiresExplicitFinishedEvidence() throws Exception {
        enableBgm();
        var fixture = episode(false);
        String id = compose.submitCompose(fixture.episode(), 0, id(), USER, true);
        var claim = jobs.claim("missing-receipt").orElseThrow().claim();
        jobs.prepareSubmission(claim, "music");
        context.getBean(MediaJobQueries.class).cancel(id, OWNER);
        expire(id);
        worker.runOne();
        assertThat(jobs.get(id).status()).isEqualTo("NEEDS_REVIEW");
        assertThat(audio.reserved(id)).isTrue();
        var request = review(id, EpisodeAudioReviewService.Action.CONFIRM_FINISHED, null, null, false);
        assertThatThrownBy(() -> reviews.review(id, request, OWNER)).hasMessageContaining("明确确认");
        var accepted = review(id, EpisodeAudioReviewService.Action.CONFIRM_FINISHED, null, null, true);
        assertThatThrownBy(() -> reviews.review(id, accepted, OTHER)).isInstanceOf(RuntimeException.class);
        reviews.review(id, accepted, OWNER);
        reviews.review(id, accepted, OWNER);
        assertThat(audio.reserved(id)).isFalse();
        worker.runOne();
        assertThat(jobs.get(id).status()).isEqualTo("CANCELLED");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_async_outbox WHERE aggregate_id = ? AND event_type = 'EPISODE_AUDIO_REVIEW'", Integer.class, id)).isEqualTo(1);
        context.getBean(GenerationOutbox.class).dispatch();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_job_event WHERE job_id = ?", Integer.class, id)).isEqualTo(1);
        verifyNoInteractions(sonilo, fetcher);
    }

    @Test void rejectedAudioSubmissionRetainsFailureAndReleasesCapacityWithoutReview() {
        var config = enableBgm();
        var fixture = episode(false);
        when(sonilo.submitMusicTask(eq(config), any(), any()))
                .thenThrow(GenerationProviderFailure.http(503, "audio queue is full"));
        String id = compose.submitCompose(fixture.episode(), 0, id(), USER, true);
        worker.runOne();
        assertThat(jobs.get(id).status()).isEqualTo("FAILED");
        assertThat(audio.result(id, "music").path("error").asText()).contains("HTTP 503", "audio queue is full");
        assertThat(audio.reserved(id)).isFalse();
        assertThat(audio.unresolved(id)).isEmpty();
        verify(sonilo, times(1)).submitMusicTask(eq(config), any(), any());
        verify(sonilo, never()).tryQueryTask(any(), any());
    }

    @Test void lateOriginalResultReleasesCapacityButCannotWriteWithExpiredLease() {
        enableBgm();
        var fixture = episode(false);
        String id = compose.submitCompose(fixture.episode(), 0, id(), USER, true);
        var claim = jobs.claim("late-original").orElseThrow().claim();
        var intent = jobs.prepareSubmission(claim, "music");
        jobs.acknowledge(intent.attempt(), "late-music");
        expire(id);
        var result = json.object(Map.of("status", "succeeded"));
        audio.received(id, "music", intent.attempt(), result);
        audio.received(id, "music", intent.attempt(), result);
        assertThat(audio.reserved(id)).isFalse();
        assertThatThrownBy(() -> audio.received(id, "music", intent.attempt(), json.object(Map.of("status", "failed"))))
                .isInstanceOf(MediaException.class);
        assertThatThrownBy(() -> completion.complete(claim)).isInstanceOf(MediaException.class);
        assertThat(jdbc.queryForObject("SELECT composed_video_url FROM afv_storyboard_episode WHERE id = ?", String.class, fixture.episode())).isNull();
    }

    @Test void sameApiBlocksSecondComposeButDoesNotBlockItsPreSubmitCancellation() {
        enableBgm();
        String first = compose.submitCompose(episode(false).episode(), 0, id(), USER, true);
        String second = compose.submitCompose(episode(false).episode(), 0, id(), USER, true);
        assertThat(jobs.claim("first-capacity").orElseThrow().id()).isEqualTo(first);
        assertThat(jobs.claim("second-capacity")).isEmpty();
        context.getBean(MediaJobQueries.class).cancel(second, OWNER);
        worker.runOne();
        assertThat(jobs.get(second).status()).isEqualTo("CANCELLED");
        assertThat(audio.reserved(first)).isTrue();
        assertThat(audio.reserved(second)).isFalse();
        verifyNoInteractions(sonilo);
    }

    @Test void sfxWaitsForNewReservationAfterMusicResultReleasedTheSlot() throws Exception {
        var config = enableBgm();
        context.getBean(EpisodeBgmProperties.class).setSfxEnabled(true);
        String id = compose.submitCompose(episode(false).episode(), 0, id(), USER, true);
        when(sonilo.submitMusicTask(eq(config), any(), any())).thenReturn("first-track");
        when(sonilo.submitSfxTask(eq(config), any(), isNull())).thenReturn("second-track");
        successfulAudio(config, "first-track");
        when(sonilo.tryQueryTask(config, "second-track")).thenReturn(JSONUtil.parseObj("{\"status\":\"failed\"}"));
        worker.runOne();
        assertThat(jobs.get(id).status()).isEqualTo("WAITING");
        assertThat(jobs.get(id).checkpoint().path("music").asText()).startsWith("media://");
        assertThat(audio.reserved(id)).isFalse();
        verify(sonilo, never()).submitSfxTask(any(), any(), any());
        due(id);
        worker.runOne();
        assertThat(jobs.get(id).status()).isEqualTo("FAILED");
        assertThat(audio.reserved(id)).isFalse();
        verify(sonilo, times(1)).submitMusicTask(eq(config), any(), any());
        verify(sonilo, times(1)).submitSfxTask(eq(config), any(), isNull());
    }

    @Test void failedIngestionResumesSavedResponseWithoutQueryOrPostAndKeepsTargetVersion() throws Exception {
        var config = enableBgm();
        var fixture = episode(false);
        String id = compose.submitCompose(fixture.episode(), 0, id(), USER, true);
        when(sonilo.submitMusicTask(eq(config), any(), any())).thenReturn("saved-response");
        successfulAudio(config, "saved-response");
        doThrow(MediaException.invalid("下载暂时失败")).when(fetcher).copyPublic(anyString(), any(), any());
        worker.runOne();
        assertThat(jobs.get(id).status()).isEqualTo("NEEDS_REVIEW");
        assertThat(audio.reserved(id)).isFalse();
        assertThat(jdbc.queryForObject("SELECT version FROM afv_storyboard_episode WHERE id = ?", Integer.class, fixture.episode())).isEqualTo(1);
        copyAudio();
        reviews.review(id, review(id, EpisodeAudioReviewService.Action.RETRY_INGEST, null, null, false), OWNER);
        worker.runOne();
        assertThat(jobs.get(id).status()).as(jobs.get(id).error()).isEqualTo("SUCCEEDED");
        verify(sonilo, times(1)).submitMusicTask(eq(config), any(), any());
        verify(sonilo, times(1)).tryQueryTask(config, "saved-response");
    }

    @Test void queryFailuresBackOffThenKeepReviewAndCapacityWithoutReposting() {
        var config = enableBgm();
        String id = compose.submitCompose(episode(false).episode(), 0, id(), USER, true);
        when(sonilo.submitMusicTask(eq(config), any(), any())).thenReturn("query-failed");
        when(sonilo.tryQueryTask(config, "query-failed")).thenThrow(new IllegalStateException("模拟查询失败"));
        for (int index = 0; index < 8; index++) {
            due(id);
            worker.runOne();
            assertThat(audio.reserved(id)).isTrue();
            assertThat(jobs.get(id).status()).isEqualTo(index == 7 ? "NEEDS_REVIEW" : "WAITING");
        }
        verify(sonilo, times(1)).submitMusicTask(eq(config), any(), any());
        verify(sonilo, times(8)).tryQueryTask(config, "query-failed");
    }

    @Test void receiptRecoveryObservesOnlyOriginalRequestAndDoesNotPermitStaleDecision() throws Exception {
        var config = enableBgm();
        String id = compose.submitCompose(episode(false).episode(), 0, id(), USER, true);
        var claim = jobs.claim("receipt-lost").orElseThrow().claim();
        jobs.prepareSubmission(claim, "music");
        expire(id);
        worker.runOne();
        var stale = review(id, EpisodeAudioReviewService.Action.CONFIRM_FINISHED, null, null, true);
        var request = review(id, EpisodeAudioReviewService.Action.RECORD_RECEIPT, "music", "restored-original", false);
        reviews.review(id, request, OWNER);
        reviews.review(id, request, OWNER);
        assertThatThrownBy(() -> reviews.review(id, stale, OWNER)).isInstanceOf(MediaException.class);
        successfulAudio(config, "restored-original");
        worker.runOne();
        assertThat(jobs.get(id).status()).as(jobs.get(id).error()).isEqualTo("SUCCEEDED");
        verify(sonilo, never()).submitMusicTask(any(), any(), any());
        verify(sonilo, times(1)).tryQueryTask(config, "restored-original");
    }

    @Test void lateReceiptWithoutRevisionChangeAndLateResultInvalidateOwnerDecision() {
        enableBgm();
        String id = compose.submitCompose(episode(false).episode(), 0, id(), USER, true);
        var claim = jobs.claim("owner-review-evidence").orElseThrow().claim();
        var intent = jobs.prepareSubmission(claim, "music");
        expire(id);
        worker.runOne();
        var beforeReceipt = review(id, EpisodeAudioReviewService.Action.CONFIRM_FINISHED, null, null, true);
        jobs.acknowledge(intent.attempt(), "original-late-receipt");
        assertThat(jobs.get(id).revision()).isEqualTo(beforeReceipt.expectedRevision());
        assertThatThrownBy(() -> reviews.review(id, beforeReceipt, OWNER)).hasMessageContaining("已变化");
        assertThat(audio.reserved(id)).isTrue();
        var beforeResult = review(id, EpisodeAudioReviewService.Action.CONFIRM_FINISHED, null, null, true);
        audio.received(id, "music", intent.attempt(), json.object(Map.of("status", "failed")));
        assertThatThrownBy(() -> reviews.review(id, beforeResult, OWNER)).hasMessageContaining("已变化");
        reviews.review(id, review(id, EpisodeAudioReviewService.Action.CONFIRM_FINISHED, null, null, true), OWNER);
        worker.runOne();
        assertThat(jobs.get(id).status()).isEqualTo("CANCELLED");
        verifyNoInteractions(sonilo, fetcher);
    }

    private EpisodeAudioReviewService.Review review(String id, EpisodeAudioReviewService.Action action, String step, String receipt, boolean ended) {
        var detail = reviews.detail(id, OWNER);
        return new EpisodeAudioReviewService.Review(detail.revision(), detail.factsHash(), action, "受控测试：已核实原供应商记录", step, receipt, ended, new GenerationClientScope(71L, null));
    }
    private void successfulAudio(ApiConfig config, String remote) throws Exception {
        var response = JSONUtil.parseObj("{\"status\":\"succeeded\"}");
        when(sonilo.tryQueryTask(config, remote)).thenReturn(response);
        when(sonilo.extractPrimaryTrack(response, remote)).thenReturn(new SoniloAudioTrack("https://audio.example.test/result"));
        copyAudio();
    }
    private void copyAudio() throws Exception {
        doAnswer(call -> { Files.copy(musicFile, call.getArgument(2, Path.class), StandardCopyOption.REPLACE_EXISTING); return null; })
                .when(fetcher).copyPublic(anyString(), any(), any());
    }
    private void due(String id) { jdbc.update("UPDATE afv_media_job SET next_run_at = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE id = ?", id); }

    private ApiConfig enableBgm() {
        var config = ApiConfig.builder().id(900L).name("测试配乐").platform("sonilo").status(1).apiKey("test-key").apiUrl("https://api.example.test").build();
        when(configs.getListByPlatform("sonilo")).thenReturn(List.of(config));
        when(configs.getById(900L)).thenReturn(config);
        return config;
    }
    private void expire(String id) { jdbc.update("UPDATE afv_media_job SET lease_until = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE id = ?", id); }
    private Fixture episode(boolean second) {
        long board = IDS.incrementAndGet(), episode = IDS.incrementAndGet(), scene = IDS.incrementAndGet(), shot = IDS.incrementAndGet();
        jdbc.update("INSERT INTO afv_storyboard(id, project_id) VALUES (?, 71)", board);
        jdbc.update("INSERT INTO afv_storyboard_episode(id, storyboard_id, episode_number, title) VALUES (?, ?, 1, '合成验证')", episode, board);
        jdbc.update("INSERT INTO afv_storyboard_scene(id, storyboard_id, episode_id) VALUES (?, ?, ?)", scene, board, episode);
        jdbc.update("INSERT INTO afv_storyboard_item(id, storyboard_id, storyboard_episode_id, storyboard_scene_id, video_url, sort_order) VALUES (?, ?, ?, ?, ?, 0)",
                shot, board, episode, scene, MediaReference.of(silentMedia));
        if (second) jdbc.update("INSERT INTO afv_storyboard_item(id, storyboard_id, storyboard_episode_id, storyboard_scene_id, video_url, sort_order) VALUES (?, ?, ?, ?, ?, 1)",
                IDS.incrementAndGet(), board, episode, scene, MediaReference.of(audioMedia));
        return new Fixture(episode, shot);
    }
    private record Fixture(long episode, long shot) { }
    private static void ffmpeg(String... args) throws Exception {
        var command = new ArrayList<>(List.of("ffmpeg", "-v", "error", "-nostdin", "-y"));
        command.addAll(List.of(args));
        Path log = Files.createTempFile(directory, "fixture-", ".log");
        Process process = new ProcessBuilder(command).redirectErrorStream(true).redirectOutput(log.toFile()).start();
        boolean done = process.waitFor(30, TimeUnit.SECONDS);
        if (!done) process.destroyForcibly();
        assertThat(done).isTrue();
        assertThat(process.exitValue()).as(Files.readString(log)).isZero();
    }
}
