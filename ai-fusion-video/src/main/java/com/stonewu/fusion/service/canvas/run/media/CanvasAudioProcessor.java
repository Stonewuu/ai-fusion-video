package com.stonewu.fusion.service.canvas.run.media;

import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.sonilo.SoniloAudioClient;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Ports;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import com.stonewu.fusion.service.media.*;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobOutputs;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaKind;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class CanvasAudioProcessor {
    private final CanvasMediaJobs tasks;
    private final MediaJobRepository jobs;
    private final MediaResourceQueries media;
    private final MediaContentReader contents;
    private final MediaJobOutputs outputs;
    private final ProviderMediaFetcher fetcher;
    private final CanvasRunSources sources;
    private final SoniloAudioClient sonilo;
    private final CanvasJson json;

    public void step(MediaJob job, CanvasMediaInput input, Path work) throws IOException {
        var original = jobs.submission(job.id(), "audio");
        if (job.cancelRequested() && original.isEmpty()) {
            tasks.finish(job.claim(), "CANCELLED", "尚未提交供应商，已停止", null, false);
            return;
        }
        if (!jobs.now().isBefore(job.deadline())) {
            boolean unknown = original.isPresent() && tasks.received(job.id()) == null;
            tasks.finish(job.claim(), unknown ? "UNKNOWN_SUBMISSION" : "FAILED",
                    unknown ? "远端结果未知，已结束音频任务的本地等待" : "音频处理超过等待期限", null, false);
            return;
        }
        var api = tasks.api(input);
        if (original.isEmpty()) {
            var value = input.inputs().at("video").getFirst();
            sources.authorize(value, job.context().actor(), true);
            var resource = media.get(value.mediaId(), job.context().actor());
            long maximum = input.operation() == CanvasNodeKind.MUSIC_GENERATION ? 360000 : 180000;
            if (resource.content().durationMs() == null || resource.content().durationMs() < 3000 || resource.content()
                    .durationMs() > maximum) {
                throw CanvasRunException.invalid("音频输入视频时长必须为 3 秒至 " + maximum / 1000 + " 秒");
            }
            Path video = work.resolve("audio-source.mp4");
            contents.copy(value.mediaId(), job.context().actor(), video, MediaIntakeService.MAX_RESULT_BYTES);
            sources.authorize(value, job.context().actor(), true);
            var intent = tasks.beginAudio(job.claim());
            if (!intent.created()) {
                return;
            }
            String prompt = input.config().path("prompt").asText("");
            String remoteId;
            try {
                remoteId = input.operation() == CanvasNodeKind.MUSIC_GENERATION
                        ? sonilo.submitMusicTask(api, video, prompt) : sonilo.submitSfxTask(api, video, prompt);
            } catch (RuntimeException failure) {
                var rejected = GenerationProviderFailure.find(failure);
                if (rejected == null) {
                    throw failure;
                }
                tasks.received(job.claim(), json.object(Map.of("status", "failed", "error", rejected.getMessage())));
                tasks.finish(job.claim(), "FAILED", rejected.getMessage(), null, false);
                return;
            }
            jobs.acknowledge(intent.attempt(), remoteId);
            tasks.waitNext(job.claim(), "供应商已受理，等待音频", false);
            return;
        }
        var result = tasks.received(job.id());
        if (original.get().providerId() == null && result == null) {
            tasks.finish(job.claim(), "UNKNOWN_SUBMISSION", "远端结果未知，未取得可查询的音频回执，原请求不会自动重发",
                    null, false);
            return;
        }
        if (result == null) {
            var response = sonilo.tryQueryTask(api, original.get().providerId());
            if (response == null) {
                tasks.waitNext(job.claim(), "查询暂不可用，稍后查询原任务", true);
                return;
            }
            String state = response.getStr("status");
            if (List.of("pending", "processing", "queued", "running").contains(state)) {
                tasks.waitNext(job.claim(), job.cancelRequested() ? "供应商不支持取消，等待原任务结束" : "等待音频生成",
                        false);
                return;
            }
            if (!"succeeded".equals(state) && !"failed".equals(state)) {
                throw CanvasRunException.invalid("供应商未返回可识别的音频状态，远端结果未知");
            }
            result = json.read(response.toString(), ObjectNode.class);
            tasks.received(job.claim(), result);
        }
        var current = jobs.observe(job.claim());
        if (current.cancelRequested()) {
            tasks.finish(job.claim(), "CANCELLED", "供应商原任务已结束，后续执行已停止", null, false);
            return;
        }
        if (result.path("status").asText().equals("failed")) {
            tasks.finish(job.claim(), "FAILED", result.path("error").asText("供应商音频生成失败"), null, false);
            return;
        }
        tasks.check(job.claim(), false);
        var previous = outputs.existing(job.claim(), "audio", MediaKind.AUDIO);
        String reference;
        if (previous.isPresent()) {
            reference = previous.get();
        } else {
            String url = sonilo.extractPrimaryTrack(JSONUtil.parseObj(result.toString()), original.get().providerId())
                    .url();
            Path file = work.resolve("generated.audio");
            fetcher.copyPublic(url, api, file);
            reference = outputs.store(job.claim(), "audio", file, "application/octet-stream", MediaKind.AUDIO);
        }
        var value = sources.mediaValue(MediaReference.id(reference), job.context().actor(), job.projectId());
        tasks.finish(job.claim(), "SUCCEEDED", null, new Ports(Map.of("audio", List.of(value))), false);
    }
}
