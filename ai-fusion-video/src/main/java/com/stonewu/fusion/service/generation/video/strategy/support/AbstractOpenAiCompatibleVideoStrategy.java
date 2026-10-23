package com.stonewu.fusion.service.generation.video.strategy.support;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.job.*;
import com.stonewu.fusion.service.generation.transport.GenerationHttpClients;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
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
    private final AiModelMetadataResolver aiModelMetadataResolver;
    private final OpenAiCompatibleVideoProtocolSupport protocolSupport;
    private final OpenAiCompatibleVideoProtocolAdapter protocolAdapter;

    private final OkHttpClient okHttpClient = GenerationHttpClients.providerRequestBuilder()
            .connectTimeout(1, TimeUnit.MINUTES)
            .readTimeout(25, TimeUnit.MINUTES)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build();

    protected AbstractOpenAiCompatibleVideoStrategy(AiModelService aiModelService,
                                                    ApiConfigService apiConfigService,
                                                    VideoGenerationService videoGenerationService,
                                                    ModelPresetService modelPresetService,
                                                    AiModelMetadataResolver aiModelMetadataResolver,
                                                    OpenAiCompatibleVideoProtocolSupport protocolSupport,
                                                    OpenAiCompatibleVideoProtocolAdapter protocolAdapter) {
        this.aiModelService = aiModelService;
        this.apiConfigService = apiConfigService;
        this.videoGenerationService = videoGenerationService;
        this.modelPresetService = modelPresetService;
        this.aiModelMetadataResolver = aiModelMetadataResolver;
        this.protocolSupport = protocolSupport;
        this.protocolAdapter = protocolAdapter;
    }

    @Override
    public PreparedGenerationCall prepare(VideoTask task, AiModel model, ApiConfig apiConfig) {
        GenerationHttpClients.requireNoTransaction();
        var context = buildProtocolContext(model, apiConfig, task, resolveModelConfig(model));
        RequestBody body = protocolAdapter.buildSubmitBody(context);
        Request request = new Request.Builder().url(protocolAdapter.resolveSubmitUrl(context))
                .addHeader("Authorization", "Bearer " + apiConfig.getApiKey()).post(body).build();
        var client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
        return new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () ->
                GenerationProviderReceipt.accepted(submitTask(context, protocolAdapter, request, client)));
    }

    @Override
    public RemoteGenerationResult query(String platformTaskId, VideoTask task, AiModel model, ApiConfig apiConfig) {
        GenerationHttpClients.requireNoTransaction();
        var config = resolveModelConfig(model);
        var context = buildProtocolContext(model, apiConfig, task, config);
        var result = queryTask(context, protocolAdapter, platformTaskId);
        return ProviderJobStates.result(protocolSupport.normalizeStatus(result.status()), result.errorMessage(),
                (int) Math.ceil(resolvePollIntervalMillis(config) / 1000.0), () -> {
                    String video = result.videoUrl();
                    if (StrUtil.isBlank(video)) {
                        video = protocolAdapter.resolveVideoContentUrl(context, platformTaskId, result);
                    }
                    String cover = result.coverUrl();
                    if (StrUtil.isBlank(cover)) {
                        cover = protocolAdapter.resolveCoverContentUrl(context, platformTaskId, result);
                    }
                    return List.of(RemoteGenerationResult.Output.video(video, cover,
                            result.durationSeconds() == null ? task.getDuration() : result.durationSeconds()));
                });
    }


    private String submitTask(OpenAiCompatibleVideoProtocolContext protocolContext,
                              OpenAiCompatibleVideoProtocolAdapter protocolAdapter, Request request,
                              OkHttpClient client) {
        GenerationHttpClients.requireNoTransaction();
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
        Integer timeoutSeconds = protocolSupport.getInteger(modelConfig, "pollTimeoutSeconds", "pollTimeout",
                "timeoutSeconds");
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


}
