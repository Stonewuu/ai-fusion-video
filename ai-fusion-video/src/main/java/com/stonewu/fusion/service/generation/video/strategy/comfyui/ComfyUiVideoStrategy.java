package com.stonewu.fusion.service.generation.video.strategy.comfyui;

import com.stonewu.fusion.entity.ai.ApiConfig;

import com.stonewu.fusion.service.generation.job.ProviderJobStates;

import com.stonewu.fusion.service.generation.job.GenerationLeaseScope;

import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;

import com.stonewu.fusion.service.generation.job.GenerationProviderReceipt;

import com.stonewu.fusion.service.generation.job.GenerationPreparedRequest;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiExecutionContext;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiGenerationExecutor;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiBoundInputs;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiPreparedSubmission;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiStoredOutput;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowService;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiJobResult;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
import com.stonewu.fusion.service.media.GenerationMediaService;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.generation.GenerationKind;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/** Video generation through an administrator-published ComfyUI workflow. */
@Component
@RequiredArgsConstructor
@Slf4j
public class ComfyUiVideoStrategy implements VideoGenerationStrategy {

    private static final long DEFAULT_POLL_INTERVAL_MILLIS = 5_000L;
    private static final long DEFAULT_TIMEOUT_MILLIS = 60L * 60L * 1_000L;

    private final AiModelService aiModelService;
    private final VideoGenerationService videoGenerationService;
    private final ComfyUiGenerationExecutor executor;
    private final GenerationMediaService media;

    @Override
    public String getName() {
        return ComfyUiWorkflowService.PLATFORM;
    }

    @Override
    public int batchSize(VideoTask task, AiModel model, ApiConfig apiConfig) { return task.getCount(); }

    @Override
    public PreparedGenerationCall prepare(VideoTask task, AiModel model, ApiConfig apiConfig) {
        var context = executor.resolveContext(model, task.getWorkflowVersionId(), apiConfig);
        var values = videoValues(task, model);
        var prepared = executor.prepare(context, GenerationLeaseScope.require().id(), values);
        return new PreparedGenerationCall(new GenerationPreparedRequest(prepared.promptId(), prepared.prompt()), () -> {
            executor.submit(prepared);
            return GenerationProviderReceipt.accepted(prepared.promptId());
        });
    }

    @Override
    public RemoteGenerationResult query(String platformTaskId, VideoTask task, AiModel model, ApiConfig apiConfig) {
        var context = executor.resolveContext(model, task.getWorkflowVersionId(), apiConfig);
        var job = executor.query(context, platformTaskId);
        if (!job.completed()) return ProviderJobStates.result(job.status(), "ComfyUI 工作流未成功结束", 3, List::of);
        String prefix = "comfy-" + GenerationLeaseScope.require().id() + "-";
        var stored = executor.storeOutputs(context, job, (output, file, index) ->
                media.file(GenerationKind.VIDEO, task.getId(), prefix + index, file.path(),
                        switch (output.mediaType()) {
                            case "image" -> MediaKind.IMAGE;
                            case "video" -> MediaKind.VIDEO;
                            default -> throw new BusinessException("ComfyUI 输出媒体类型未登记");
                        }));
        var videos = stored.stream().filter(output -> "video".equals(output.mediaType()) && "primary".equals(output.role())).toList();
        var covers = stored.stream().filter(output -> "image".equals(output.mediaType()) && "cover".equals(output.role())).toList();
        if (!covers.isEmpty() && covers.size() != videos.size()) throw new BusinessException("ComfyUI 视频和封面数量不匹配");
        var results = new ArrayList<RemoteGenerationResult.Output>();
        for (int i = 0; i < videos.size(); i++) {
            results.add(RemoteGenerationResult.Output.video(videos.get(i).url(), covers.isEmpty() ? null : covers.get(i).url(), task.getDuration()));
        }
        return RemoteGenerationResult.success(results);
    }

    @Override
    public boolean cancel(String platformTaskId, VideoTask task, AiModel model, ApiConfig apiConfig) {
        return executor.cancel(executor.resolveContext(model, task.getWorkflowVersionId(), apiConfig), platformTaskId);
    }





    private Map<String, Object> videoValues(VideoTask task, AiModel model) {
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("prompt", StrUtil.blankToDefault(task.getPrompt(), ""));
        values.put("duration", task.getDuration());
        values.put("count", Math.max(1, task.getCount()));
        values.put("seed", task.getSeed() != null && task.getSeed() >= 0 ? task.getSeed() : randomSeed());
        values.put("generateAudio", task.getGenerateAudio());
        int[] resolution = parseResolution(task.getResolution());
        if (resolution != null) {
            values.put("width", resolution[0]);
            values.put("height", resolution[1]);
        }
        Integer fps = configInt(model, "defaultFps");
        if (fps != null) values.put("fps", fps);
        if (StrUtil.isNotBlank(task.getFirstFrameImageUrl())) {
            values.put("firstFrame", task.getFirstFrameImageUrl());
        }
        if (StrUtil.isNotBlank(task.getLastFrameImageUrl())) {
            values.put("lastFrame", task.getLastFrameImageUrl());
        }
        List<String> images = parseList(task.getReferenceImageUrls());
        if (!images.isEmpty()) values.put("referenceImages", images);
        List<String> videos = parseList(task.getReferenceVideoUrls());
        if (!videos.isEmpty()) values.put("referenceVideos", videos);
        List<String> audios = parseList(task.getReferenceAudioUrls());
        if (!audios.isEmpty()) values.put("referenceAudios", audios);
        ComfyUiBoundInputs.overlay(values, task.getWorkflowInputs(), task.getCount());
        return values;
    }

    private List<String> parseList(String json) {
        if (StrUtil.isBlank(json)) return List.of();
        try {
            return new ArrayList<>(JSONUtil.parseArray(json).toList(String.class));
        } catch (RuntimeException e) {
            throw new BusinessException(400, "ComfyUI 参考素材列表不是合法 JSON 数组");
        }
    }

    private int[] parseResolution(String resolution) {
        if (StrUtil.isBlank(resolution) || !resolution.matches("\\d{2,5}x\\d{2,5}")) return null;
        String[] parts = resolution.split("x", 2);
        return new int[]{Integer.parseInt(parts[0]), Integer.parseInt(parts[1])};
    }

    private long pollInterval(AiModel model) {
        return configLong(model, "comfyuiPollIntervalMillis", DEFAULT_POLL_INTERVAL_MILLIS);
    }

    private long timeout(AiModel model) {
        return configLong(model, "comfyuiTimeoutMillis", DEFAULT_TIMEOUT_MILLIS);
    }

    private Integer configInt(AiModel model, String key) {
        if (model == null || StrUtil.isBlank(model.getConfig())) return null;
        try {
            return JSONUtil.parseObj(model.getConfig()).getInt(key);
        } catch (RuntimeException ignored) {
            return null;
        }
    }

    private long configLong(AiModel model, String key, long fallback) {
        if (model == null || StrUtil.isBlank(model.getConfig())) return fallback;
        try {
            JSONObject config = JSONUtil.parseObj(model.getConfig());
            Long value = config.getLong(key);
            return value != null && value > 0 ? value : fallback;
        } catch (RuntimeException ignored) {
            return fallback;
        }
    }

    private long randomSeed() {
        return ThreadLocalRandom.current().nextLong(Long.MAX_VALUE);
    }
}
