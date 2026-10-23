package com.stonewu.fusion.service.media.job;

import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.sonilo.SoniloAudioClient;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import com.stonewu.fusion.service.media.*;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.storyboard.EpisodeBgmService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

@Component
@RequiredArgsConstructor
public class ComposeBgmProcessor {
    private final CanvasJson json;
    private final MediaJobRepository jobs;
    private final EpisodeAudioJobs audio;
    private final MediaJobOutputs outputs;
    private final MediaContentReader content;
    private final MediaResourceQueries resources;
    private final ApiConfigService configs;
    private final SoniloAudioClient sonilo;
    private final ProviderMediaFetcher fetcher;
    private final EpisodeBgmService mixer;
    private final MediaProcessRunner processes;

    /**
     * 每轮至多查询每个外部任务一次，未就绪时保存下次执行时间并释放工作者。
     */
    public boolean process(MediaJob.Claim claim, Path work) throws IOException {
        var job = jobs.check(claim);
        var checkpoint = job.checkpoint().deepCopy();
        var settings = json.read(json.write(job.input()), ComposeSnapshot.class).bgm();
        if (checkpoint.hasNonNull("video")) {
            return true;
        }
        if (settings == null) {
            checkpoint.put("video", checkpoint.path("baseVideo").asText());
            jobs.checkpoint(claim, checkpoint, "准备写回成片");
            return true;
        }
        ApiConfig config = api(settings);
        var base = resources.get(MediaReference.id(checkpoint.path("baseVideo").asText()), job.context().actor());
        double duration = base.content().durationMs() / 1000.0;
        String unsuitable = mixer.musicSkipReason(duration);
        if (unsuitable != null) {
            throw MediaException.invalid(unsuitable + "，无配乐成片已保留");
        }
        if (settings.sfx() && !mixer.sfxAllowed(duration)) {
            throw MediaException.invalid("本集时长超过音效上限 3 分钟，无配乐成片已保留");
        }
        Path video = work.resolve("music-source.mp4");
        content.copy(base.id(), job.context().actor(), video, MediaIntakeService.MAX_RESULT_BYTES);
        if (!track(claim, settings, config, "music", video, work)) {
            return false;
        }
        if (settings.sfx() && !track(claim, settings, config, "sfx", video, work)) {
            return false;
        }
        job = jobs.check(claim);
        checkpoint = job.checkpoint().deepCopy();
        var existing = outputs.existing(claim, "mixed", MediaKind.VIDEO);
        if (existing.isPresent()) {
            checkpoint.put("video", existing.get());
        } else {
            Path music = work.resolve("music.audio");
            content.copy(MediaReference.id(checkpoint.path("music").asText()), job.context().actor(), music,
                    MediaIntakeService.MAX_RESULT_BYTES);
            Path sfx = null;
            if (settings.sfx()) {
                sfx = work.resolve("sfx.audio");
                content.copy(MediaReference.id(checkpoint.path("sfx").asText()), job.context().actor(), sfx,
                        MediaIntakeService.MAX_RESULT_BYTES);
            }
            Path mixed = work.resolve("mixed.mp4");
            processes.run(mixer.buildMixCommand(video, music, sfx, base.content().hasAudio(), duration, mixed,
                            settings.volume()),
                    work, mixed, () -> jobs.check(claim));
            checkpoint.put("video", outputs.store(claim, "mixed", mixed, "video/mp4", MediaKind.VIDEO));
        }
        jobs.checkpoint(claim, checkpoint, "配乐混音完成，准备写回");
        return true;
    }

    /**
     * 取消后仅观察已经发出的调用；不准备视频、不生成下一轨，也不下载或写回新产物。
     */
    public void observeCancellation(MediaJob.Claim claim) {
        var job = jobs.observe(claim);
        var pending = audio.unresolved(job.id());
        if (pending.isEmpty()) {
            throw new MediaException(409, "MEDIA_COMPOSE_CANCELLED", "原音频请求已结束，后续处理已停止");
        }
        var settings = json.read(json.write(job.input()), ComposeSnapshot.class).bgm();
        var config = api(settings);
        for (String step : pending) {
            if (poll(claim, settings, config, step) == null) {
                return;
            }
        }
        throw new MediaException(409, "MEDIA_COMPOSE_CANCELLED", "供应商原任务已结束，后续处理已停止");
    }

    private boolean track(MediaJob.Claim claim, ComposeSnapshot.Bgm settings, ApiConfig config, String step, Path video,
                          Path work) throws IOException {
        var job = jobs.check(claim);
        var previousResponse = audio.result(job.id(), step);
        if (previousResponse != null && previousResponse.path("status").asText().equals("failed")) {
            throw new MediaException(409, "MEDIA_AUDIO_PROVIDER_FAILED",
                    previousResponse.path("error").asText("供应商音频生成失败"));
        }
        if (job.checkpoint().hasNonNull(step)) {
            return true;
        }
        var existing = outputs.existing(claim, step, MediaKind.AUDIO);
        if (existing.isPresent()) {
            var checkpoint = job.checkpoint().deepCopy().put(step, existing.get());
            jobs.checkpoint(claim, checkpoint, step.equals("music") ? "配乐已就绪" : "音效已就绪");
            return true;
        }
        if (jobs.submission(job.id(), step).isEmpty() && !audio.reserved(job.id())) {
            audio.waitNext(claim, "等待下一轨音频的并发额度", 1, false);
            return false;
        }
        var submission = jobs.prepareSubmission(claim, step);
        String providerId = submission.providerId();
        if (providerId == null) {
            if (!submission.created()) {
                throw unknown();
            }
            jobs.check(claim);
            try {
                providerId = step.equals("music") ? sonilo.submitMusicTask(config, video,
                        settings.prompt()) : sonilo.submitSfxTask(config, video, null);
                jobs.acknowledge(submission.attempt(), providerId);
            } catch (RuntimeException failure) {
                var rejected = GenerationProviderFailure.find(failure);
                if (rejected != null) {
                    audio.rejected(job.id(), step, submission.attempt(), rejected);
                    throw new MediaException(409, "MEDIA_AUDIO_PROVIDER_FAILED", rejected.getMessage());
                }
                var error = unknown();
                error.initCause(failure);
                throw error;
            }
        }
        ObjectNode response = poll(claim, settings, config, step);
        if (response == null) {
            return false;
        }
        jobs.check(claim);
        if ("failed".equals(response.path("status").asText())) {
            throw new MediaException(409, "MEDIA_AUDIO_PROVIDER_FAILED",
                    step.equals("music") ? "供应商配乐生成失败" : "供应商音效生成失败");
        }
        String url = sonilo.extractPrimaryTrack(JSONUtil.parseObj(response.toString()), providerId).url();
        Path file = work.resolve(step + ".result");
        fetcher.copyPublic(url, config, file);
        String reference = outputs.store(claim, step, file, "application/octet-stream", MediaKind.AUDIO);
        var checkpoint = jobs.check(claim).checkpoint().deepCopy().put(step, reference);
        jobs.checkpoint(claim, checkpoint, step.equals("music") ? "配乐已就绪" : "音效已就绪");
        return true;
    }

    private ObjectNode poll(MediaJob.Claim claim, ComposeSnapshot.Bgm settings, ApiConfig config, String step) {
        var job = jobs.observe(claim);
        var response = audio.result(job.id(), step);
        if (response != null) {
            return response;
        }
        var submission = jobs.submission(job.id(), step).orElseThrow(MediaJobFence::lost);
        if (submission.providerId() == null) {
            throw unknown();
        }
        if (!jobs.now().isBefore(job.deadline()) || !jobs.now()
                .isBefore(audio.pollDeadline(job.id(), step, settings.timeoutSeconds()))) {
            throw new MediaException(409, "MEDIA_PROVIDER_WAIT_EXPIRED", "原音频请求超过观察时限，请核实供应商任务");
        }
        try {
            var queried = sonilo.tryQueryTask(config, submission.providerId());
            if (queried == null) {
                throw new IllegalStateException("查询没有返回有效结果");
            }
            response = json.read(queried.toString(), ObjectNode.class);
        } catch (RuntimeException failure) {
            var error = new MediaException(409, "MEDIA_AUDIO_QUERY_FAILED", "原音频请求查询暂时失败");
            error.initCause(failure);
            throw error;
        }
        String status = response.path("status").asText();
        if (List.of("succeeded", "failed").contains(status)) {
            audio.received(job.id(), step, submission.attempt(), response);
            return response;
        }
        if (!List.of("pending", "processing", "queued", "running").contains(status)) {
            throw new MediaException(409, "MEDIA_AUDIO_STATUS_UNKNOWN", "供应商音频状态未知，请核实原调用");
        }
        audio.waitNext(claim, job.cancelRequested() ? "供应商不支持取消，等待原音频请求结束"
                : step.equals("music") ? "等待配乐生成" : "等待音效生成", settings.pollSeconds(), false);
        return null;
    }

    private ApiConfig api(ComposeSnapshot.Bgm settings) {
        ApiConfig config = configs.getById(settings.apiConfigId());
        if (config == null || !Integer.valueOf(1).equals(config.getStatus()) || !json.hash(config)
                .equals(settings.configHash())) {
            throw new MediaException(409, "MEDIA_AUDIO_CONFIG_CHANGED", "原配乐服务配置已变化，请先核实已有调用");
        }
        return config;
    }

    private MediaException unknown() {
        return new MediaException(409, "MEDIA_SUBMISSION_UNKNOWN",
                "付费请求是否受理尚未确认，已停止自动重发，请先核对供应商记录");
    }
}
