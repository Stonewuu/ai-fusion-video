package com.stonewu.fusion.service.generation.strategy.impl;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.generation.VideoGenerationService;
import com.stonewu.fusion.service.generation.strategy.VideoGenerationStrategy;
import com.volcengine.ark.runtime.model.content.generation.CreateContentGenerationTaskRequest;
import com.volcengine.ark.runtime.model.content.generation.CreateContentGenerationTaskResult;
import com.volcengine.ark.runtime.model.content.generation.GetContentGenerationTaskRequest;
import com.volcengine.ark.runtime.model.content.generation.GetContentGenerationTaskResponse;
import com.volcengine.ark.runtime.service.ArkService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * 火山引擎（方舟）Seedance 2.0 视频生成策略
 * <p>
 * 使用 volcengine-java-sdk-ark-runtime 的 ArkService 调用 Content Generation API。
 * 支持文生视频、首帧图生视频、首尾帧图生视频、多模态参考等能力。
 * <p>
 * API 为异步模式：先创建任务获取 taskId，然后轮询任务状态直到成功或失败。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class VolcengineVideoStrategy implements VideoGenerationStrategy {

    /** 默认模型 ID：Seedance 2.0 */
    private static final String DEFAULT_MODEL_ID = "doubao-seedance-2-0-260128";

    /** 轮询间隔（秒） */
    private static final int POLL_INTERVAL_SECONDS = 10;

    /** 默认轮询超时（秒） */
    private static final int DEFAULT_POLL_TIMEOUT_SECONDS = 3600;

    /** 最大轮询次数（10秒 × 360次 = 3600秒 = 60分钟） */
    private static final int MAX_POLL_COUNT = DEFAULT_POLL_TIMEOUT_SECONDS / POLL_INTERVAL_SECONDS;

    private final AiModelService aiModelService;
    private final ApiConfigService apiConfigService;
    private final VideoGenerationService videoGenerationService;

    @Override
    public String getName() {
        return "volcengine";
    }

    @Override
    public String submit(VideoTask task) {
        AiModel model = resolveModel(task);
        ApiConfig apiConfig = resolveApiConfig(model);
        String modelCode = resolveModelCode(model);

        ArkService service = buildArkService(apiConfig);
        try {
            // 构建 content 列表
            List<CreateContentGenerationTaskRequest.Content> contents = buildContents(task);

            // 构建创建任务请求
            CreateContentGenerationTaskRequest.Builder reqBuilder = CreateContentGenerationTaskRequest.builder()
                    .model(modelCode)
                    .content(contents)
                    .watermark(task.getWatermark() != null ? task.getWatermark() : false)
                    .generateAudio(task.getGenerateAudio() != null ? task.getGenerateAudio() : false)
                    .cameraFixed(task.getCameraFixed() != null ? task.getCameraFixed() : false);

            // 设置画面比例
            if (StrUtil.isNotBlank(task.getRatio())) {
                reqBuilder.ratio(task.getRatio());
            }

            // 设置分辨率
            if (StrUtil.isNotBlank(task.getResolution())) {
                reqBuilder.resolution(task.getResolution());
            }

            // 设置视频时长
            if (task.getDuration() != null && task.getDuration() > 0) {
                reqBuilder.duration(task.getDuration().longValue());
            }

            // 设置随机种子
            if (task.getSeed() != null) {
                reqBuilder.seed(task.getSeed());
            }

            // 返回视频尾帧（便于后续延长视频）
            reqBuilder.returnLastFrame(true);

            CreateContentGenerationTaskRequest request = reqBuilder.build();

            log.info("[Volcengine Video] 创建视频生成任务: model={}, mode={}, ratio={}, duration={}s",
                    modelCode, task.getGenerateMode(), task.getRatio(), task.getDuration());

            CreateContentGenerationTaskResult result = service.createContentGenerationTask(request);
            String platformTaskId = result.getId();

            log.info("[Volcengine Video] 任务已创建: platformTaskId={}", platformTaskId);
            return platformTaskId;
        } finally {
            service.shutdownExecutor();
        }
    }

    @Override
    public void poll(String platformTaskId, VideoTask task) {
        AiModel model = resolveModel(task);
        ApiConfig apiConfig = resolveApiConfig(model);

        ArkService service = buildArkService(apiConfig);
        try {
            GetContentGenerationTaskRequest getRequest = GetContentGenerationTaskRequest.builder()
                    .taskId(platformTaskId)
                    .build();

            log.info("[Volcengine Video] 开始轮询任务状态: platformTaskId={}", platformTaskId);

            for (int i = 0; i < MAX_POLL_COUNT; i++) {
                GetContentGenerationTaskResponse response = service.getContentGenerationTask(getRequest);
                String status = response.getStatus();

                if ("succeeded".equalsIgnoreCase(status)) {
                    log.info("[Volcengine Video] 任务成功: platformTaskId={}", platformTaskId);
                    handleSuccess(response, task);
                    return;
                } else if ("failed".equalsIgnoreCase(status)) {
                    String errorMsg = "未知错误";
                    if (response.getError() != null) {
                        errorMsg = response.getError().getMessage();
                        log.error("[Volcengine Video] 任务失败: platformTaskId={}, code={}, message={}",
                                platformTaskId, response.getError().getCode(), errorMsg);
                    }
                    throw new RuntimeException("火山引擎视频生成失败: " + errorMsg);
                } else {
                    log.debug("[Volcengine Video] 任务进行中: platformTaskId={}, status={}, pollCount={}/{}",
                            platformTaskId, status, i + 1, MAX_POLL_COUNT);
                    try {
                        TimeUnit.SECONDS.sleep(POLL_INTERVAL_SECONDS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("视频生成轮询被中断", e);
                    }
                }
            }

            throw new RuntimeException("火山引擎视频生成超时（轮询 " + MAX_POLL_COUNT + " 次）");
        } finally {
            service.shutdownExecutor();
        }
    }

    // ========== 内部辅助方法 ==========

    /**
     * 构建 Content 列表（文本提示词 + 参考图片 + 参考视频 + 参考音频）
     */
    private List<CreateContentGenerationTaskRequest.Content> buildContents(VideoTask task) {
        List<CreateContentGenerationTaskRequest.Content> contents = new ArrayList<>();

        // 1. 文本提示词（必须）
        if (StrUtil.isNotBlank(task.getPrompt())) {
            contents.add(CreateContentGenerationTaskRequest.Content.builder()
                    .type("text")
                    .text(task.getPrompt())
                    .build());
        }

        // 2. 首帧参考图片（图生视频模式）
        if (StrUtil.isNotBlank(task.getFirstFrameImageUrl())) {
            contents.add(CreateContentGenerationTaskRequest.Content.builder()
                    .type("image_url")
                    .imageUrl(CreateContentGenerationTaskRequest.ImageUrl.builder()
                            .url(task.getFirstFrameImageUrl())
                            .build())
                    .role("first_frame")
                    .build());
        }

        // 3. 尾帧参考图片
        if (StrUtil.isNotBlank(task.getLastFrameImageUrl())) {
            contents.add(CreateContentGenerationTaskRequest.Content.builder()
                    .type("image_url")
                    .imageUrl(CreateContentGenerationTaskRequest.ImageUrl.builder()
                            .url(task.getLastFrameImageUrl())
                            .build())
                    .role("last_frame")
                    .build());
        }

        // 4. 额外参考图片（多模态参考场景，最多 9 张）
        List<String> refImageUrls = parseJsonUrls(task.getReferenceImageUrls());
        if (refImageUrls != null) {
            for (String refUrl : refImageUrls) {
                contents.add(CreateContentGenerationTaskRequest.Content.builder()
                        .type("image_url")
                        .imageUrl(CreateContentGenerationTaskRequest.ImageUrl.builder()
                                .url(refUrl)
                                .build())
                        .role("reference_image")
                        .build());
            }
        }

        // 5. 参考视频（编辑视频 / 延长视频 / 多模态参考，最多 3 个）
        List<String> refVideoUrls = parseJsonUrls(task.getReferenceVideoUrls());
        if (refVideoUrls != null) {
            for (String refUrl : refVideoUrls) {
                contents.add(CreateContentGenerationTaskRequest.Content.builder()
                        .type("video_url")
                        .videoUrl(CreateContentGenerationTaskRequest.VideoUrl.builder()
                                .url(refUrl)
                                .build())
                        .role("reference_video")
                        .build());
            }
        }

        // 6. 参考音频（有声视频生成，最多 3 个）
        List<String> refAudioUrls = parseJsonUrls(task.getReferenceAudioUrls());
        if (refAudioUrls != null) {
            for (String refUrl : refAudioUrls) {
                contents.add(CreateContentGenerationTaskRequest.Content.builder()
                        .type("audio_url")
                        .audioUrl(CreateContentGenerationTaskRequest.AudioUrl.builder()
                                .url(refUrl)
                                .build())
                        .role("reference_audio")
                        .build());
            }
        }

        return contents;
    }

    /**
     * 处理任务成功：提取视频 URL 并更新数据库
     */
    private void handleSuccess(GetContentGenerationTaskResponse response, VideoTask task) {
        String videoUrl = null;
        String lastFrameUrl = null;

        if (response.getContent() != null) {
            videoUrl = response.getContent().getVideoUrl();
            lastFrameUrl = response.getContent().getLastFrameUrl();
        }

        if (StrUtil.isBlank(videoUrl)) {
            throw new RuntimeException("火山引擎返回成功但无视频 URL");
        }

        // 更新 VideoItem
        List<VideoItem> items = videoGenerationService.listItems(task.getId());
        if (!items.isEmpty()) {
            VideoItem item = items.get(0);
            item.setVideoUrl(videoUrl);
            item.setLastFrameUrl(lastFrameUrl);
            item.setStatus(1); // 成功
            if (response.getDuration() != null) {
                item.setDuration(response.getDuration().intValue());
            }
            videoGenerationService.updateItem(item);
        }

        task.setSuccessCount(1);
        videoGenerationService.update(task);

        log.info("[Volcengine Video] 视频生成完成: taskId={}, videoUrl={}, lastFrame={}",
                task.getTaskId(), videoUrl, lastFrameUrl != null ? "有" : "无");
    }

    /**
     * 构建 ArkService 实例
     */
    private ArkService buildArkService(ApiConfig apiConfig) {
        ArkService.Builder builder = ArkService.builder().apiKey(apiConfig.getApiKey());
        if (StrUtil.isNotBlank(apiConfig.getApiUrl())) {
            builder.baseUrl(apiConfig.getApiUrl());
        } else {
            builder.baseUrl("https://ark.cn-beijing.volces.com/api/v3");
        }
        return builder.build();
    }

    /**
     * 解析模型代码
     */
    private String resolveModelCode(AiModel model) {
        if (model != null && StrUtil.isNotBlank(model.getCode())) {
            return model.getCode();
        }
        return DEFAULT_MODEL_ID;
    }

    /**
     * 从 VideoTask 解析关联的 AiModel
     */
    private AiModel resolveModel(VideoTask task) {
        if (task.getModelId() != null) {
            try {
                return aiModelService.getById(task.getModelId());
            } catch (Exception e) {
                log.warn("[Volcengine Video] 获取模型失败: modelId={}", task.getModelId());
            }
        }
        return null;
    }

    /**
     * 从 AiModel 解析关联的 ApiConfig
     */
    private ApiConfig resolveApiConfig(AiModel model) {
        if (model != null && model.getApiConfigId() != null) {
            try {
                ApiConfig config = apiConfigService.getById(model.getApiConfigId());
                if (config != null) {
                    return config;
                }
            } catch (Exception e) {
                log.warn("[Volcengine Video] 获取 API 配置失败: apiConfigId={}", model.getApiConfigId());
            }
        }
        throw new RuntimeException("未找到火山引擎视频生成 API 配置，请在系统设置中配置 volcengine 平台 API Key");
    }

    /**
     * 解析 JSON 数组字符串为 URL 列表
     *
     * @param jsonUrls JSON 数组字符串，如 ["url1","url2"]，可为 null
     * @return URL 列表，无内容时返回 null
     */
    private List<String> parseJsonUrls(String jsonUrls) {
        if (StrUtil.isBlank(jsonUrls)) {
            return null;
        }
        try {
            JSONArray arr = JSONUtil.parseArray(jsonUrls);
            List<String> urls = arr.toList(String.class);
            return urls.isEmpty() ? null : urls;
        } catch (Exception e) {
            return null;
        }
    }
}
