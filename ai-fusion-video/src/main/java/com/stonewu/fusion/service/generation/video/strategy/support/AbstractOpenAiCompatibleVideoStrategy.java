package com.stonewu.fusion.service.generation.video.strategy.support;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
import com.stonewu.fusion.service.storage.MediaStorageService;
import lombok.extern.slf4j.Slf4j;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * OpenAI 兼容视频生成策略（Sora Videos API）。
 * <p>
 * 对接官方接口：
 * POST /v1/videos                 创建视频生成任务（multipart/form-data）
 * GET  /v1/videos/{id}            查询任务状态
 * GET  /v1/videos/{id}/content    下载生成的视频内容
 * <p>
 * 支持文生视频与图生视频（input_reference 参考图）。视频内容需带 Authorization 鉴权下载，
 * 因此在本策略内主动拉取字节并落地到持久化存储，VideoItem.videoUrl 直接保存为本地可访问地址。
 */
@Slf4j
public abstract class AbstractOpenAiCompatibleVideoStrategy implements VideoGenerationStrategy {

    private static final long DEFAULT_POLL_INTERVAL_MILLIS = 10000L;
    private static final long DEFAULT_POLL_TIMEOUT_MILLIS = 60L * 60L * 1000L;

    private final AiModelService aiModelService;
    private final ApiConfigService apiConfigService;
    private final VideoGenerationService videoGenerationService;
    private final ModelPresetService modelPresetService;
    private final MediaStorageService mediaStorageService;
    private final AiModelMetadataResolver aiModelMetadataResolver;
    private final OpenAiCompatibleVideoProtocolSupport protocolSupport;
    private final OpenAiCompatibleVideoProtocolAdapter protocolAdapter;

    private final OkHttpClient okHttpClient = new OkHttpClient.Builder()
            .connectTimeout(1, TimeUnit.MINUTES)
            .readTimeout(25, TimeUnit.MINUTES)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build();

    protected AbstractOpenAiCompatibleVideoStrategy(AiModelService aiModelService,
                                                     ApiConfigService apiConfigService,
                                                     VideoGenerationService videoGenerationService,
                                                     ModelPresetService modelPresetService,
                                                     MediaStorageService mediaStorageService,
                                                     AiModelMetadataResolver aiModelMetadataResolver,
                                                     OpenAiCompatibleVideoProtocolSupport protocolSupport,
                                                     OpenAiCompatibleVideoProtocolAdapter protocolAdapter) {
        this.aiModelService = aiModelService;
        this.apiConfigService = apiConfigService;
        this.videoGenerationService = videoGenerationService;
        this.modelPresetService = modelPresetService;
        this.mediaStorageService = mediaStorageService;
        this.aiModelMetadataResolver = aiModelMetadataResolver;
        this.protocolSupport = protocolSupport;
        this.protocolAdapter = protocolAdapter;
    }

    @Override
    public String submit(VideoTask task) {
        AiModel model = resolveModel(task);
        ApiConfig apiConfig = resolveApiConfig(model);
        JSONObject modelConfig = resolveModelConfig(model);
        OpenAiCompatibleVideoProtocolContext protocolContext = buildProtocolContext(model, apiConfig, task, modelConfig);

        List<VideoItem> items = videoGenerationService.listItems(task.getId());
        if (items.isEmpty()) {
            throw new BusinessException("视频任务缺少生成条目");
        }

        String firstPlatformTaskId = null;
        for (VideoItem item : items) {
            if (StrUtil.isNotBlank(item.getPlatformTaskId())) {
                firstPlatformTaskId = StrUtil.blankToDefault(firstPlatformTaskId, item.getPlatformTaskId());
                continue;
            }

            String platformTaskId = submitTask(protocolContext, protocolAdapter);
            item.setPlatformTaskId(platformTaskId);
            videoGenerationService.updateItem(item);

            if (firstPlatformTaskId == null) {
                firstPlatformTaskId = platformTaskId;
            }
        }

        log.info("[{} Video] 任务已创建: taskId={}, model={}, count={}",
                providerLabel(), task.getTaskId(), protocolSupport.resolveModelCode(model), items.size());
        return firstPlatformTaskId;
    }

    @Override
    public void poll(String platformTaskId, VideoTask task) {
        AiModel model = resolveModel(task);
        ApiConfig apiConfig = resolveApiConfig(model);
        JSONObject modelConfig = resolveModelConfig(model);
        OpenAiCompatibleVideoProtocolContext protocolContext = buildProtocolContext(model, apiConfig, task, modelConfig);

        List<VideoItem> items = videoGenerationService.listItems(task.getId());
        if (items.isEmpty()) {
            OpenAiCompatibleVideoTaskResult result = waitForTask(protocolContext, protocolAdapter, platformTaskId);
            persistResult(protocolContext, protocolAdapter, platformTaskId, result, null, task);
            task.setSuccessCount(1);
            videoGenerationService.update(task);
            return;
        }

        int successCount = 0;
        for (VideoItem item : items) {
            String currentPlatformTaskId = StrUtil.blankToDefault(item.getPlatformTaskId(), platformTaskId);
            if (StrUtil.isBlank(currentPlatformTaskId)) {
                item.setStatus(2);
                item.setErrorMsg(providerLabel() + " 平台任务 ID 为空");
                videoGenerationService.updateItem(item);
                throw new BusinessException(providerLabel() + " 平台任务 ID 为空");
            }

            try {
                OpenAiCompatibleVideoTaskResult result = waitForTask(protocolContext, protocolAdapter, currentPlatformTaskId);
                persistResult(protocolContext, protocolAdapter, currentPlatformTaskId, result, item, task);
                item.setPlatformTaskId(currentPlatformTaskId);
                item.setStatus(1);
                item.setErrorMsg(null);
                videoGenerationService.updateItem(item);
                successCount++;
            } catch (BusinessException e) {
                item.setStatus(2);
                item.setErrorMsg(e.getMessage());
                videoGenerationService.updateItem(item);
                throw e;
            }
        }

        task.setSuccessCount(successCount);
        videoGenerationService.update(task);
        log.info("[{} Video] 视频生成完成: taskId={}, successCount={}",
                providerLabel(), task.getTaskId(), successCount);
    }

    private void persistResult(OpenAiCompatibleVideoProtocolContext protocolContext,
                               OpenAiCompatibleVideoProtocolAdapter protocolAdapter,
                               String trackingId,
                               OpenAiCompatibleVideoTaskResult result,
                               VideoItem item,
                               VideoTask task) {
        String videoUrl = result.videoUrl();
        String coverUrl = result.coverUrl();

        if (StrUtil.isBlank(videoUrl)) {
            String contentUrl = protocolAdapter.resolveVideoContentUrl(protocolContext, trackingId, result);
            if (StrUtil.isBlank(contentUrl)) {
                throw new BusinessException(providerLabel() + " 视频任务成功但未获取到视频内容: " + trackingId);
            }
            videoUrl = downloadVideoContent(protocolContext.apiConfig(), contentUrl);
        }

        if (StrUtil.isBlank(coverUrl)) {
            String coverContentUrl = protocolAdapter.resolveCoverContentUrl(protocolContext, trackingId, result);
            if (StrUtil.isNotBlank(coverContentUrl)) {
                coverUrl = downloadThumbnail(protocolContext.apiConfig(), coverContentUrl);
            }
        }

        if (item != null) {
            item.setVideoUrl(videoUrl);
            item.setCoverUrl(coverUrl);
            item.setDuration(result.durationSeconds() != null ? result.durationSeconds() : task.getDuration());
        }
    }

    private String submitTask(OpenAiCompatibleVideoProtocolContext protocolContext,
                              OpenAiCompatibleVideoProtocolAdapter protocolAdapter) {
        RequestBody requestBody = protocolAdapter.buildSubmitBody(protocolContext);
        Request request = new Request.Builder()
                .url(protocolAdapter.resolveSubmitUrl(protocolContext))
                .addHeader("Authorization", "Bearer " + protocolContext.apiConfig().getApiKey())
                .post(requestBody)
                .build();

        OkHttpClient client = AiProxySupport.okHttpClient(okHttpClient, protocolContext.apiConfig());
        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body() != null ? response.body().string() : "";
            if (!response.isSuccessful()) {
                throw new BusinessException(providerLabel() + " 视频任务提交失败: HTTP " + response.code() + " "
                        + protocolSupport.extractErrorMessage(responseBody));
            }
            OpenAiCompatibleVideoTaskResult result = protocolAdapter.parseSubmitResponse(protocolContext, responseBody);
            if (StrUtil.isBlank(result.trackingId())) {
                throw new BusinessException(providerLabel() + " 视频任务未返回跟踪 ID");
            }
            return result.trackingId();
        } catch (IOException e) {
            throw new BusinessException(providerLabel() + " 视频任务提交异常: " + e.getMessage());
        }
    }

    private OpenAiCompatibleVideoTaskResult waitForTask(OpenAiCompatibleVideoProtocolContext protocolContext,
                                                        OpenAiCompatibleVideoProtocolAdapter protocolAdapter,
                                                        String platformTaskId) {
        JSONObject modelConfig = protocolContext.modelConfig();
        long pollIntervalMillis = resolvePollIntervalMillis(modelConfig);
        long deadline = System.currentTimeMillis() + resolvePollTimeoutMillis(modelConfig);

        while (System.currentTimeMillis() <= deadline) {
            OpenAiCompatibleVideoTaskResult result = queryTask(protocolContext, protocolAdapter, platformTaskId);
            String status = protocolSupport.normalizeStatus(result.status());

            if ("completed".equals(status) || "succeeded".equals(status) || "success".equals(status)) {
                return result;
            }
            if ("failed".equals(status) || "error".equals(status)
                    || "canceled".equals(status) || "cancelled".equals(status)) {
                throw new BusinessException(providerLabel() + " 视频任务失败: "
                        + StrUtil.blankToDefault(result.errorMessage(), "未知错误"));
            }

            sleepQuietly(pollIntervalMillis);
        }

        throw new BusinessException(providerLabel() + " 视频任务轮询超时: " + platformTaskId);
    }

    private OpenAiCompatibleVideoTaskResult queryTask(OpenAiCompatibleVideoProtocolContext protocolContext,
                                                      OpenAiCompatibleVideoProtocolAdapter protocolAdapter,
                                                      String platformTaskId) {
        Request request = new Request.Builder()
                .url(protocolAdapter.resolveQueryUrl(protocolContext, platformTaskId))
                .addHeader("Authorization", "Bearer " + protocolContext.apiConfig().getApiKey())
                .get()
                .build();

        OkHttpClient client = AiProxySupport.okHttpClient(okHttpClient, protocolContext.apiConfig());
        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body() != null ? response.body().string() : "";
            if (!response.isSuccessful()) {
                throw new BusinessException(providerLabel() + " 视频任务查询失败: HTTP " + response.code() + " "
                        + protocolSupport.extractErrorMessage(responseBody));
            }
            return protocolAdapter.parseQueryResponse(protocolContext, responseBody);
        } catch (IOException e) {
            throw new BusinessException(providerLabel() + " 视频任务查询异常: " + e.getMessage());
        }
    }

    private String downloadVideoContent(ApiConfig apiConfig, String contentUrl) {
        Request request = new Request.Builder()
                .url(contentUrl)
                .addHeader("Authorization", "Bearer " + apiConfig.getApiKey())
                .addHeader("Accept", "video/*,*/*;q=0.8")
                .get()
                .build();

        OkHttpClient client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful() || response.body() == null) {
                String body = response.body() != null ? response.body().string() : "";
                throw new BusinessException(providerLabel() + " 视频内容下载失败: HTTP " + response.code() + " "
                        + protocolSupport.extractErrorMessage(body));
            }
            byte[] bytes = response.body().bytes();
            String extension = protocolSupport.extensionFromVideoMimeType(response.header("Content-Type"));
            return mediaStorageService.storeBytes(bytes, "videos", extension);
        } catch (IOException e) {
            throw new BusinessException(providerLabel() + " 视频内容下载异常: " + e.getMessage());
        }
    }

    private String downloadThumbnail(ApiConfig apiConfig, String thumbnailUrl) {
        Request request = new Request.Builder()
                .url(thumbnailUrl)
                .addHeader("Authorization", "Bearer " + apiConfig.getApiKey())
                .addHeader("Accept", "image/*,*/*;q=0.8")
                .get()
                .build();

        OkHttpClient client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful() || response.body() == null) {
                return null;
            }
            byte[] bytes = response.body().bytes();
            if (bytes.length == 0) {
                return null;
            }
            String extension = protocolSupport.extensionFromImageMimeType(response.header("Content-Type"));
            return mediaStorageService.storeBytes(bytes, "images", extension);
        } catch (Exception e) {
            log.warn("[{} Video] 视频封面下载失败（忽略）: url={}, error={}",
                    providerLabel(), thumbnailUrl, e.getMessage());
            return null;
        }
    }

    private AiModel resolveModel(VideoTask task) {
        if (task.getModelId() == null) {
            throw new BusinessException(providerLabel() + " 视频任务缺少 modelId");
        }
        AiModel model = aiModelService.getById(task.getModelId());
        if (model == null) {
            throw new BusinessException(providerLabel() + " 视频模型不存在: modelId=" + task.getModelId());
        }
        return model;
    }

    private ApiConfig resolveApiConfig(AiModel model) {
        if (model.getApiConfigId() == null) {
            throw new BusinessException(providerLabel() + " 视频模型缺少 apiConfigId");
        }
        ApiConfig apiConfig = apiConfigService.getById(model.getApiConfigId());
        if (apiConfig == null) {
            throw new BusinessException(providerLabel() + " API 配置不存在");
        }
        if (StrUtil.isBlank(apiConfig.getApiKey())) {
            throw new BusinessException(providerLabel() + " 缺少 API Key 配置");
        }
        return apiConfig;
    }

    private String resolveModelCode(AiModel model) {
        return protocolSupport.resolveModelCode(model);
    }

    private JSONObject resolveModelConfig(AiModel model) {
        JSONObject merged = new JSONObject();
        String capabilityPresetCode = model != null ? model.getCapabilityPresetCode() : null;
        if (modelPresetService != null && StrUtil.isNotBlank(capabilityPresetCode)) {
            mergeConfig(merged, parseConfig(modelPresetService.getPresetConfig(capabilityPresetCode)));
        }
        if (model != null) {
            mergeConfig(merged, parseConfig(model.getConfig()));
        }
        return merged;
    }

    private JSONObject parseConfig(String configJson) {
        if (StrUtil.isBlank(configJson)) {
            return null;
        }
        try {
            return JSONUtil.parseObj(configJson);
        } catch (Exception e) {
            log.warn("[{} Video] 模型配置解析失败，已忽略附加参数: config={}",
                    providerLabel(), configJson);
            return null;
        }
    }

    private void mergeConfig(JSONObject target, JSONObject source) {
        if (target == null || source == null || source.isEmpty()) {
            return;
        }
        for (String key : source.keySet()) {
            target.set(key, source.get(key));
        }
    }

    private long resolvePollIntervalMillis(JSONObject modelConfig) {
        Long interval = protocolSupport.getLong(modelConfig, "pollIntervalMillis", "pollIntervalMs", "pollInterval");
        return interval != null && interval > 0 ? interval : DEFAULT_POLL_INTERVAL_MILLIS;
    }

    private long resolvePollTimeoutMillis(JSONObject modelConfig) {
        Long timeoutMillis = protocolSupport.getLong(modelConfig, "pollTimeoutMillis", "pollTimeoutMs");
        if (timeoutMillis != null && timeoutMillis > 0) {
            return timeoutMillis;
        }
        Integer timeoutSeconds = protocolSupport.getInteger(modelConfig, "pollTimeoutSeconds", "pollTimeout", "timeoutSeconds");
        if (timeoutSeconds != null && timeoutSeconds > 0) {
            return timeoutSeconds * 1000L;
        }
        return DEFAULT_POLL_TIMEOUT_MILLIS;
    }

    private OpenAiCompatibleVideoProtocolContext buildProtocolContext(AiModel model,
                                                                     ApiConfig apiConfig,
                                                                     VideoTask task,
                                                                     JSONObject modelConfig) {
        return new OpenAiCompatibleVideoProtocolContext(
                model,
                apiConfig,
                task,
                modelConfig,
                aiModelMetadataResolver.resolve(model, apiConfig)
        );
    }

    protected String providerLabel() {
        return getName();
    }

    private void sleepQuietly(long intervalMillis) {
        try {
            TimeUnit.MILLISECONDS.sleep(intervalMillis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new BusinessException(providerLabel() + " 视频任务轮询被中断");
        }
    }
}
