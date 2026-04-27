package com.stonewu.fusion.service.generation.strategy.impl;

import com.alibaba.cloud.ai.dashscope.api.DashScopeVideoApi;
import com.alibaba.cloud.ai.dashscope.spec.DashScopeApiSpec.VideoGenerationRequest;
import com.alibaba.cloud.ai.dashscope.spec.DashScopeApiSpec.VideoGenerationResponse;
import com.alibaba.dashscope.aigc.videosynthesis.VideoSynthesis;
import com.alibaba.dashscope.aigc.videosynthesis.VideoSynthesisParam;
import com.alibaba.dashscope.aigc.videosynthesis.VideoSynthesisResult;
import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.dashscope.DashScopeGenerationSupport;
import com.stonewu.fusion.service.generation.VideoGenerationService;
import com.stonewu.fusion.service.generation.strategy.VideoGenerationStrategy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * 阿里云百炼 DashScope 视频生成策略。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DashScopeVideoStrategy implements VideoGenerationStrategy {

    private static final long POLL_INTERVAL_MILLIS = 15000L;
    private static final int MAX_POLL_COUNT = 400;

    private final AiModelService aiModelService;
    private final ApiConfigService apiConfigService;
    private final VideoGenerationService videoGenerationService;
    private final DashScopeGenerationSupport dashScopeSupport;

    @Override
    public String getName() {
        return DashScopeGenerationSupport.PLATFORM;
    }

    @Override
    public String submit(VideoTask task) {
        AiModel model = resolveModel(task);
        ApiConfig apiConfig = resolveApiConfig(model);
        DashScopeGenerationSupport.requireApiKey(apiConfig);
        String modelCode = model != null && StrUtil.isNotBlank(model.getCode()) ? model.getCode() : "wan2.7-t2v";
        JSONObject config = DashScopeGenerationSupport.parseModelConfig(model);

        List<VideoItem> items = videoGenerationService.listItems(task.getId());
        if (items.isEmpty()) {
            throw new BusinessException("视频任务缺少生成条目");
        }

        boolean useSdk = shouldUseSpringAiVideoSdk(modelCode, task);
        DashScopeVideoApi videoApi = useSdk ? buildSdkVideoApi(apiConfig) : null;
        String firstPlatformTaskId = null;
        for (VideoItem item : items) {
            if (StrUtil.isNotBlank(item.getPlatformTaskId())) {
                firstPlatformTaskId = StrUtil.blankToDefault(firstPlatformTaskId, item.getPlatformTaskId());
                continue;
            }
            String taskId;
            if (useSdk) {
                taskId = submitWithSdk(videoApi, buildSdkVideoRequest(modelCode, task, config));
            } else {
                // Spring AI Alibaba 1.1.2.0 的 Video SDK 还不能表达 Wan2.7 media 数组、ratio/watermark 等新协议参数；
                // DashScope Java SDK 2.22.x 已提供 VideoSynthesis 高层 API，可兼容这些模型且无需手写 HTTP。
                taskId = submitWithDashScopeJavaSdk(apiConfig, buildOfficialVideoParam(apiConfig, modelCode, task, config));
            }
            if (StrUtil.isBlank(taskId)) {
                throw new BusinessException("DashScope 视频任务未返回 task_id");
            }
            item.setPlatformTaskId(taskId);
            videoGenerationService.updateItem(item);
            if (firstPlatformTaskId == null) {
                firstPlatformTaskId = taskId;
            }
            log.info("[DashScope Video] 任务已创建: taskId={}, platformTaskId={}, model={}",
                    task.getTaskId(), taskId, modelCode);
        }
        return firstPlatformTaskId;
    }

    @Override
    public void poll(String platformTaskId, VideoTask task) {
        AiModel model = resolveModel(task);
        ApiConfig apiConfig = resolveApiConfig(model);
        DashScopeGenerationSupport.requireApiKey(apiConfig);
        String modelCode = model != null && StrUtil.isNotBlank(model.getCode()) ? model.getCode() : "wan2.7-t2v";
        boolean useSdk = shouldUseSpringAiVideoSdk(modelCode, task);
        DashScopeVideoApi videoApi = useSdk ? buildSdkVideoApi(apiConfig) : null;

        List<VideoItem> items = videoGenerationService.listItems(task.getId());
        int successCount = 0;
        if (items.isEmpty()) {
            DashScopeVideoResult result = useSdk
                    ? waitForSdkVideoTask(videoApi, platformTaskId)
                    : waitForOfficialVideoTask(apiConfig, platformTaskId);
            if (StrUtil.isBlank(result.videoUrl())) {
                throw new BusinessException("DashScope 视频任务成功但未返回视频地址: " + platformTaskId);
            }
            task.setSuccessCount(1);
            videoGenerationService.update(task);
            return;
        }

        for (VideoItem item : items) {
            String currentPlatformTaskId = StrUtil.blankToDefault(item.getPlatformTaskId(), platformTaskId);
            if (StrUtil.isBlank(currentPlatformTaskId)) {
                item.setStatus(2);
                item.setErrorMsg("DashScope 平台任务 ID 为空");
                videoGenerationService.updateItem(item);
                continue;
            }
            DashScopeVideoResult result = useSdk
                    ? waitForSdkVideoTask(videoApi, currentPlatformTaskId)
                    : waitForOfficialVideoTask(apiConfig, currentPlatformTaskId);
            if (StrUtil.isBlank(result.videoUrl())) {
                item.setStatus(2);
                item.setErrorMsg("DashScope 返回成功但无视频 URL");
                videoGenerationService.updateItem(item);
                throw new BusinessException("DashScope 返回成功但无视频 URL: " + currentPlatformTaskId);
            }
            item.setPlatformTaskId(currentPlatformTaskId);
            item.setVideoUrl(result.videoUrl());
            item.setCoverUrl(result.coverUrl());
            item.setFirstFrameUrl(result.firstFrameUrl());
            item.setLastFrameUrl(result.lastFrameUrl());
            item.setDuration(result.duration() != null ? result.duration() : task.getDuration());
            item.setStatus(1);
            videoGenerationService.updateItem(item);
            successCount++;
        }

        task.setSuccessCount(successCount);
        videoGenerationService.update(task);
        log.info("[DashScope Video] 视频生成完成: taskId={}, successCount={}", task.getTaskId(), successCount);
    }

    private DashScopeVideoApi buildSdkVideoApi(ApiConfig apiConfig) {
        return DashScopeVideoApi.builder()
                .baseUrl(DashScopeGenerationSupport.resolveRootBaseUrl(apiConfig.getApiUrl()))
                .apiKey(apiConfig.getApiKey())
                .build();
    }

    private String submitWithSdk(DashScopeVideoApi videoApi, VideoGenerationRequest request) {
        VideoGenerationResponse response = videoApi.submitVideoGenTask(request).getBody();
        String taskId = extractTaskId(response);
        if (StrUtil.isBlank(taskId)) {
            throw new BusinessException("DashScope 视频 SDK 未返回 task_id: " + response);
        }
        return taskId;
    }

    private VideoGenerationRequest buildSdkVideoRequest(String modelCode, VideoTask task, JSONObject config) {
        VideoGenerationRequest.VideoInput.Builder inputBuilder = VideoGenerationRequest.VideoInput.builder();
        if (StrUtil.isNotBlank(task.getPrompt())) {
            inputBuilder.prompt(task.getPrompt());
        }
        String negativePrompt = firstNonBlank(
                DashScopeGenerationSupport.getString(config, "negativePrompt"),
                DashScopeGenerationSupport.getString(config, "negative_prompt"));
        if (StrUtil.isNotBlank(negativePrompt)) {
            inputBuilder.negativePrompt(negativePrompt);
        }

        String firstFrame = firstNonBlank(task.getFirstFrameImageUrl(), firstUrl(task.getReferenceImageUrls()));
        if (StrUtil.isNotBlank(firstFrame)) {
            inputBuilder.firstFrameUrl(toMediaUrl(firstFrame));
        }
        if (StrUtil.isNotBlank(task.getLastFrameImageUrl())) {
            inputBuilder.lastFrameUrl(toMediaUrl(task.getLastFrameImageUrl()));
        }

        VideoGenerationRequest.VideoParameters.Builder parameterBuilder = VideoGenerationRequest.VideoParameters.builder();
        String resolution = DashScopeGenerationSupport.normalizeResolutionLabel(task.getResolution());
        if (StrUtil.isNotBlank(resolution)) {
            if (resolution.matches("\\d+[xX*]\\d+")) {
                parameterBuilder.size(resolution.replace('x', '*').replace('X', '*'));
            } else {
                parameterBuilder.resolution(resolution);
            }
        } else {
            String defaultResolution = DashScopeGenerationSupport.getString(config, "defaultResolution");
            if (StrUtil.isNotBlank(defaultResolution)) {
                parameterBuilder.resolution(defaultResolution);
            }
        }
        if (task.getDuration() != null && task.getDuration() > 0) {
            parameterBuilder.duration(task.getDuration());
        } else {
            Integer defaultDuration = DashScopeGenerationSupport.getInteger(config, "defaultDuration");
            if (defaultDuration != null) {
                parameterBuilder.duration(defaultDuration);
            }
        }
        if (task.getSeed() != null) {
            parameterBuilder.seed(task.getSeed());
        } else {
            Long seed = DashScopeGenerationSupport.getLong(config, "seed");
            if (seed != null) {
                parameterBuilder.seed(seed);
            }
        }
        if (config.containsKey("promptExtend")) {
            parameterBuilder.promptExtend(DashScopeGenerationSupport.getBoolean(config, "promptExtend", false));
        } else if (config.containsKey("prompt_extend")) {
            parameterBuilder.promptExtend(DashScopeGenerationSupport.getBoolean(config, "prompt_extend", false));
        }

        return VideoGenerationRequest.builder()
                .model(modelCode)
                .input(inputBuilder.build())
                .parameters(parameterBuilder.build())
                .build();
    }

    private DashScopeVideoResult waitForSdkVideoTask(DashScopeVideoApi videoApi, String taskId) {
        for (int i = 0; i < MAX_POLL_COUNT; i++) {
            VideoGenerationResponse response = videoApi.queryVideoGenTask(taskId).getBody();
            String status = extractTaskStatus(response);
            if ("SUCCEEDED".equalsIgnoreCase(status)) {
                return extractVideoResult(response);
            }
            if ("FAILED".equalsIgnoreCase(status) || "CANCELED".equalsIgnoreCase(status)
                    || "UNKNOWN".equalsIgnoreCase(status)) {
                throw new BusinessException("DashScope 视频 SDK 任务失败: status=" + status + ", message="
                        + extractErrorMessage(response));
            }
            sleepQuietly(POLL_INTERVAL_MILLIS);
        }
        throw new BusinessException("DashScope 视频 SDK 任务轮询超时: " + taskId);
    }

    private String submitWithDashScopeJavaSdk(ApiConfig apiConfig, VideoSynthesisParam param) {
        try {
            VideoSynthesisResult response = buildOfficialVideoSynthesis(apiConfig).asyncCall(param);
            String taskId = extractTaskId(response);
            if (StrUtil.isBlank(taskId)) {
                throw new BusinessException("DashScope Java SDK 视频任务未返回 task_id: " + response);
            }
            return taskId;
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            throw new BusinessException("DashScope Java SDK 视频接口调用异常: " + e.getMessage());
        }
    }

    private VideoSynthesisParam buildOfficialVideoParam(ApiConfig apiConfig, String modelCode, VideoTask task,
                                                        JSONObject config) {
        VideoSynthesisParam.VideoSynthesisParamBuilder<?, ?> builder = VideoSynthesisParam.builder();
        builder.apiKey(apiConfig.getApiKey());
        builder.model(modelCode);
        if (StrUtil.isNotBlank(task.getPrompt())) {
            builder.prompt(task.getPrompt());
        }
        String negativePrompt = firstNonBlank(
                DashScopeGenerationSupport.getString(config, "negativePrompt"),
                DashScopeGenerationSupport.getString(config, "negative_prompt"));
        if (StrUtil.isNotBlank(negativePrompt)) {
            builder.negativePrompt(negativePrompt);
        }

        String code = modelCode.toLowerCase(Locale.ROOT);
        if (isLegacyKeyframeModel(code)) {
            String firstFrame = firstNonBlank(task.getFirstFrameImageUrl(), firstUrl(task.getReferenceImageUrls()));
            if (StrUtil.isNotBlank(firstFrame)) {
                builder.firstFrameUrl(toMediaUrl(firstFrame));
            }
            if (StrUtil.isNotBlank(task.getLastFrameImageUrl())) {
                builder.lastFrameUrl(toMediaUrl(task.getLastFrameImageUrl()));
            }
        } else if (isLegacyFirstFrameModel(code)) {
            String firstFrame = firstNonBlank(task.getFirstFrameImageUrl(), firstUrl(task.getReferenceImageUrls()));
            if (StrUtil.isNotBlank(firstFrame)) {
                builder.imgUrl(toMediaUrl(firstFrame));
            }
            String firstAudio = firstUrl(task.getReferenceAudioUrls());
            if (StrUtil.isNotBlank(firstAudio)) {
                builder.audioUrl(toMediaUrl(firstAudio));
            }
        } else {
            List<VideoSynthesisParam.Media> media = buildWan27Media(code, task);
            if (!media.isEmpty()) {
                builder.media(media);
            }
        }
        applyOfficialVideoParameters(builder, task, config);
        return builder.build();
    }

    private List<VideoSynthesisParam.Media> buildWan27Media(String modelCode, VideoTask task) {
        List<VideoSynthesisParam.Media> media = new ArrayList<>();
        List<String> referenceAudios = prepareMediaUrls(DashScopeGenerationSupport.parseJsonUrls(task.getReferenceAudioUrls()));
        int audioIndex = 0;

        addMediaIfPresent(media, "first_frame", task.getFirstFrameImageUrl(), null);
        addMediaIfPresent(media, "last_frame", task.getLastFrameImageUrl(), null);

        List<String> referenceImages = prepareMediaUrls(DashScopeGenerationSupport.parseJsonUrls(task.getReferenceImageUrls()));
        List<String> referenceVideos = prepareMediaUrls(DashScopeGenerationSupport.parseJsonUrls(task.getReferenceVideoUrls()));

        boolean isReferenceVideoModel = modelCode.contains("r2v");
        boolean isVideoEditModel = modelCode.contains("videoedit");
        boolean isImageToVideoModel = modelCode.contains("i2v");

        for (String imageUrl : referenceImages) {
            VideoSynthesisParam.Media item = mediaItem("reference_image", imageUrl);
            if (isReferenceVideoModel && audioIndex < referenceAudios.size()) {
                item.setReferenceVoice(referenceAudios.get(audioIndex++));
            }
            media.add(item);
        }

        for (String videoUrl : referenceVideos) {
            String type = isVideoEditModel || isImageToVideoModel ? "video" : "reference_video";
            VideoSynthesisParam.Media item = mediaItem(type, videoUrl);
            if (isReferenceVideoModel && audioIndex < referenceAudios.size()) {
                item.setReferenceVoice(referenceAudios.get(audioIndex++));
            }
            media.add(item);
        }

        for (; audioIndex < referenceAudios.size(); audioIndex++) {
            media.add(mediaItem("driving_audio", referenceAudios.get(audioIndex)));
        }
        return media;
    }

    private void applyOfficialVideoParameters(VideoSynthesisParam.VideoSynthesisParamBuilder<?, ?> builder,
                                              VideoTask task, JSONObject config) {
        String resolution = firstNonBlank(
                DashScopeGenerationSupport.normalizeResolutionLabel(task.getResolution()),
                DashScopeGenerationSupport.getString(config, "defaultResolution"));
        if (StrUtil.isNotBlank(resolution)) {
            if (resolution.matches("\\d+[xX*]\\d+")) {
                builder.size(resolution.replace('x', '*').replace('X', '*'));
            } else {
                builder.resolution(resolution);
            }
        }
        if (StrUtil.isNotBlank(task.getRatio())) {
            builder.ratio(task.getRatio());
        }
        if (task.getDuration() != null && task.getDuration() > 0) {
            builder.duration(task.getDuration());
        } else {
            Integer defaultDuration = DashScopeGenerationSupport.getInteger(config, "defaultDuration");
            if (defaultDuration != null) {
                builder.duration(defaultDuration);
            }
        }
        if (task.getWatermark() != null) {
            builder.watermark(task.getWatermark());
        } else if (config.containsKey("watermark")) {
            builder.watermark(DashScopeGenerationSupport.getBoolean(config, "watermark", false));
        }
        if (task.getGenerateAudio() != null) {
            builder.audio(task.getGenerateAudio());
        } else if (config.containsKey("generateAudio")) {
            builder.audio(DashScopeGenerationSupport.getBoolean(config, "generateAudio", true));
        }
        if (task.getCameraFixed() != null) {
            builder.parameter("camera_fixed", task.getCameraFixed());
        }
        if (task.getSeed() != null) {
            Integer seed = toIntegerSeed(task.getSeed());
            if (seed != null) {
                builder.seed(seed);
            }
        } else {
            Long seed = DashScopeGenerationSupport.getLong(config, "seed");
            Integer intSeed = toIntegerSeed(seed);
            if (intSeed != null) {
                builder.seed(intSeed);
            }
        }
        if (config.containsKey("promptExtend")) {
            builder.promptExtend(DashScopeGenerationSupport.getBoolean(config, "promptExtend", false));
        } else if (config.containsKey("prompt_extend")) {
            builder.promptExtend(DashScopeGenerationSupport.getBoolean(config, "prompt_extend", false));
        }
    }

    private VideoSynthesis buildOfficialVideoSynthesis(ApiConfig apiConfig) {
        return new VideoSynthesis(DashScopeGenerationSupport.resolveGenerationBaseUrl(apiConfig));
    }

    private DashScopeVideoResult waitForOfficialVideoTask(ApiConfig apiConfig, String taskId) {
        try {
            VideoSynthesis videoSynthesis = buildOfficialVideoSynthesis(apiConfig);
            for (int i = 0; i < MAX_POLL_COUNT; i++) {
                VideoSynthesisResult response = videoSynthesis.fetch(taskId, apiConfig.getApiKey());
                String status = extractTaskStatus(response);
                if ("SUCCEEDED".equalsIgnoreCase(status)) {
                    return extractVideoResult(response);
                }
                if ("FAILED".equalsIgnoreCase(status) || "CANCELED".equalsIgnoreCase(status)
                        || "UNKNOWN".equalsIgnoreCase(status)) {
                    throw new BusinessException("DashScope Java SDK 视频任务失败: status=" + status + ", message="
                            + extractErrorMessage(response));
                }
                sleepQuietly(POLL_INTERVAL_MILLIS);
            }
            throw new BusinessException("DashScope Java SDK 视频任务轮询超时: " + taskId);
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            throw new BusinessException("DashScope Java SDK 视频任务查询异常: " + e.getMessage());
        }
    }

    private DashScopeVideoResult extractVideoResult(VideoGenerationResponse response) {
        if (response == null || response.getOutput() == null) {
            return new DashScopeVideoResult(null, null, null, null, null);
        }
        Integer duration = response.getUsage() != null ? response.getUsage().getVideoDuration() : null;
        return new DashScopeVideoResult(
                response.getOutput().getVideoUrl(),
                null,
                null,
                null,
                duration
        );
    }

    private DashScopeVideoResult extractVideoResult(VideoSynthesisResult response) {
        if (response == null || response.getOutput() == null) {
            return new DashScopeVideoResult(null, null, null, null, null);
        }
        Integer duration = response.getUsage() != null ? response.getUsage().getVideoDuration() : null;
        return new DashScopeVideoResult(
                response.getOutput().getVideoUrl(),
                null,
                null,
                null,
                duration
        );
    }

    private String extractTaskId(VideoGenerationResponse response) {
        return response != null && response.getOutput() != null ? response.getOutput().getTaskId() : null;
    }

    private String extractTaskId(VideoSynthesisResult response) {
        return response != null && response.getOutput() != null ? response.getOutput().getTaskId() : null;
    }

    private String extractTaskStatus(VideoGenerationResponse response) {
        return response != null && response.getOutput() != null ? response.getOutput().getTaskStatus() : null;
    }

    private String extractTaskStatus(VideoSynthesisResult response) {
        return response != null && response.getOutput() != null ? response.getOutput().getTaskStatus() : null;
    }

    private String extractErrorMessage(VideoGenerationResponse response) {
        if (response == null || response.getOutput() == null) {
            return "";
        }
        String code = response.getOutput().getCode();
        String message = response.getOutput().getMessage();
        if (StrUtil.isNotBlank(code) && StrUtil.isNotBlank(message)) {
            return code + ": " + message;
        }
        return StrUtil.blankToDefault(message, response.toString());
    }

    private String extractErrorMessage(VideoSynthesisResult response) {
        if (response == null) {
            return "";
        }
        String code = response.getCode();
        String message = response.getMessage();
        if (StrUtil.isBlank(code) && response.getOutput() != null) {
            code = response.getOutput().getCode();
        }
        if (StrUtil.isBlank(message) && response.getOutput() != null) {
            message = response.getOutput().getMessage();
        }
        if (StrUtil.isNotBlank(code) && StrUtil.isNotBlank(message)) {
            return code + ": " + message;
        }
        return StrUtil.blankToDefault(message, response.toString());
    }

    private Integer toIntegerSeed(Long seed) {
        if (seed == null || seed < Integer.MIN_VALUE || seed > Integer.MAX_VALUE) {
            return null;
        }
        return seed.intValue();
    }

    private void addMediaIfPresent(List<VideoSynthesisParam.Media> media, String type, String url, String referenceVoice) {
        if (StrUtil.isBlank(url)) {
            return;
        }
        VideoSynthesisParam.Media item = mediaItem(type, toMediaUrl(url));
        if (StrUtil.isNotBlank(referenceVoice)) {
            item.setReferenceVoice(referenceVoice);
        }
        media.add(item);
    }

    private VideoSynthesisParam.Media mediaItem(String type, String url) {
        return VideoSynthesisParam.Media.builder()
                .type(type)
                .url(url)
                .build();
    }

    private List<String> prepareMediaUrls(List<String> urls) {
        if (urls == null || urls.isEmpty()) {
            return List.of();
        }
        List<String> result = new ArrayList<>();
        for (String url : urls) {
            if (StrUtil.isNotBlank(url)) {
                result.add(toMediaUrl(url));
            }
        }
        return result;
    }

    private String toMediaUrl(String url) {
        try {
            return dashScopeSupport.toMediaUrl(url);
        } catch (IOException e) {
            throw new BusinessException("DashScope 参考媒体处理失败: " + e.getMessage());
        }
    }

    private String firstUrl(String jsonUrls) {
        List<String> urls = DashScopeGenerationSupport.parseJsonUrls(jsonUrls);
        return urls.isEmpty() ? null : urls.get(0);
    }

    private boolean isLegacyFirstFrameModel(String modelCode) {
        String code = modelCode.toLowerCase(Locale.ROOT);
        return !code.startsWith("wan2.7-") && code.contains("i2v");
    }

    private boolean isLegacyKeyframeModel(String modelCode) {
        String code = modelCode.toLowerCase(Locale.ROOT);
        return code.contains("kf2v");
    }

    private boolean shouldUseSpringAiVideoSdk(String modelCode, VideoTask task) {
        String code = modelCode.toLowerCase(Locale.ROOT);
        if (code.contains("wan2.7-i2v") || code.contains("wan2.7-r2v") || code.contains("wan2.7-videoedit")) {
            return false;
        }
        if (StrUtil.isNotBlank(task.getRatio()) || Boolean.TRUE.equals(task.getWatermark())
            || Boolean.TRUE.equals(task.getGenerateAudio()) || Boolean.TRUE.equals(task.getCameraFixed())) {
            return false;
        }
        if (!DashScopeGenerationSupport.parseJsonUrls(task.getReferenceVideoUrls()).isEmpty()
                || !DashScopeGenerationSupport.parseJsonUrls(task.getReferenceAudioUrls()).isEmpty()) {
            return false;
        }
        List<String> referenceImages = DashScopeGenerationSupport.parseJsonUrls(task.getReferenceImageUrls());
        if (referenceImages.isEmpty()) {
            return true;
        }
        return referenceImages.size() == 1
                && StrUtil.isBlank(task.getFirstFrameImageUrl())
                && (code.contains("i2v") || code.contains("kf2v"));
    }

    private AiModel resolveModel(VideoTask task) {
        if (task != null && task.getModelId() != null) {
            try {
                return aiModelService.getById(task.getModelId());
            } catch (Exception e) {
                log.warn("[DashScope Video] 获取模型失败: modelId={}", task.getModelId(), e);
            }
        }
        return null;
    }

    private ApiConfig resolveApiConfig(AiModel model) {
        if (model != null && model.getApiConfigId() != null) {
            ApiConfig config = apiConfigService.getById(model.getApiConfigId());
            if (config != null) {
                return config;
            }
        }
        throw new BusinessException("未找到 DashScope 视频生成 API 配置，请在系统设置中配置 dashscope 平台 API Key");
    }

    private String firstNonBlank(String... values) {
        if (values == null) {
            return null;
        }
        for (String value : values) {
            if (StrUtil.isNotBlank(value)) {
                return value;
            }
        }
        return null;
    }

    private void sleepQuietly(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new BusinessException("DashScope 视频任务轮询被中断");
        }
    }

    private record DashScopeVideoResult(String videoUrl, String coverUrl, String firstFrameUrl,
                                        String lastFrameUrl, Integer duration) {
    }
}
