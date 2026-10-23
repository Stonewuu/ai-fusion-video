package com.stonewu.fusion.service.generation.image.strategy.support;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategy;
import com.stonewu.fusion.service.generation.job.GenerationPreparedRequest;
import com.stonewu.fusion.service.generation.job.GenerationProviderReceipt;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.transport.GenerationHttpClients;
import com.stonewu.fusion.service.media.GenerationMediaService;
import lombok.extern.slf4j.Slf4j;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * OpenAI-compatible image strategy.
 *
 * <p>The strategy owns task orchestration, transport and persistence only.
 * Request-shape differences live in protocol-specific packages such as
 * {@code strategy.openai}, {@code strategy.newapi} and {@code strategy.agnes}.</p>
 */
@Slf4j
public abstract class AbstractOpenAiCompatibleImageStrategy implements ImageGenerationStrategy {

    private final ImageGenerationService imageGenerationService;
    private final AiModelService aiModelService;
    private final OpenAiCompatibleImageProtocolSupport protocolSupport;
    private final OpenAiCompatibleImageProtocolAdapter protocolAdapter;
    private final GenerationMediaService generationMediaService;
    private final OkHttpClient okHttpClient;

    protected AbstractOpenAiCompatibleImageStrategy(ImageGenerationService imageGenerationService,
                                                    AiModelService aiModelService,
                                                    OpenAiCompatibleImageProtocolSupport protocolSupport,
                                                    OpenAiCompatibleImageProtocolAdapter protocolAdapter,
                                                    GenerationMediaService generationMediaService) {
        this.imageGenerationService = imageGenerationService;
        this.aiModelService = aiModelService;
        this.protocolSupport = protocolSupport;
        this.protocolAdapter = protocolAdapter;
        this.generationMediaService = generationMediaService;
        this.okHttpClient = defaultHttpClient();
    }


    @Override
    public int batchSize(ImageTask task, AiModel model, ApiConfig apiConfig) {
        var config = protocolSupport.resolveModelConfig(model.getCode(), model);
        int[] size = protocolSupport.resolveConfiguredSize(task, config);
        var context = buildContext(model, apiConfig, model.getCode(), task.getPrompt(), size[0], size[1],
                task.getCount(), parseRefImageUrls(task.getRefImageUrls()), config);
        return protocolAdapter.isSingleImagePerRequest(context) ? 1 : task.getCount();
    }

    @Override
    public PreparedGenerationCall prepare(ImageTask task, AiModel model, ApiConfig apiConfig) {
        validateApiConfig(apiConfig);
        JSONObject config = protocolSupport.resolveModelConfig(model.getCode(), model);
        int[] size = protocolSupport.resolveConfiguredSize(task, config);
        var context = buildContext(model, apiConfig, model.getCode(), task.getPrompt(), size[0], size[1],
                task.getCount(), parseRefImageUrls(task.getRefImageUrls()), config);
        var protocolRequest = protocolAdapter.buildRequest(context);
        var request = new Request.Builder().url(protocolRequest.url())
                .addHeader("Authorization", "Bearer " + apiConfig.getApiKey()).post(protocolRequest.body()).build();
        var client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
        return new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () -> {
            String response = execute(request, client, context.asyncMode() ? "异步图片任务提交" : "图片生成");
            if (context.asyncMode()) {
                return GenerationProviderReceipt.accepted(protocolSupport.parseAsyncTaskId(response));
            }
            return GenerationProviderReceipt.completed(
                    RemoteGenerationResult.success(protocolAdapter.parseImageUrls(context, response, url -> url)
                            .stream().map(RemoteGenerationResult.Output::image).toList()));
        });
    }

    @Override
    public RemoteGenerationResult query(String platformTaskId, ImageTask task, AiModel model, ApiConfig apiConfig) {
        validateApiConfig(apiConfig);
        JSONObject config = protocolSupport.resolveModelConfig(model.getCode(), model);
        if (!protocolSupport.isAsyncMode(config)) {
            throw new BusinessException("同步图片结果必须从已保存回执读取");
        }
        var result = fetchAsyncTaskResult(platformTaskId, apiConfig, config);
        if (result.completed()) {
            return RemoteGenerationResult.success(
                    result.urls().stream().map(RemoteGenerationResult.Output::image).toList());
        }
        if (result.failed()) {
            return RemoteGenerationResult.failed(result.errorMessage());
        }
        return RemoteGenerationResult.pending(
                (int) Math.ceil(protocolSupport.resolveAsyncPollIntervalMillis(config) / 1000.0));
    }


    private OpenAiCompatibleImageAsyncTaskResult fetchAsyncTaskResult(String platformTaskId,
                                                                      ApiConfig apiConfig,
                                                                      JSONObject modelConfig) {
        String requestUrl = protocolSupport.resolveAsyncTaskUrl(apiConfig, modelConfig, platformTaskId);
        Request request = new Request.Builder()
                .url(requestUrl)
                .addHeader("Authorization", "Bearer " + apiConfig.getApiKey())
                .get()
                .build();
        OkHttpClient client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body() != null ? response.body().string() : "";
            if (!response.isSuccessful()) {
                throw new RuntimeException(providerLabel() + " 异步图片任务查询失败: HTTP " + response.code()
                        + (StrUtil.isNotBlank(responseBody) ? " - " + responseBody : ""));
            }
            return protocolSupport.parseAsyncTaskResult(responseBody);
        } catch (IOException e) {
            throw new RuntimeException(providerLabel() + " 异步图片任务查询异常: " + e.getMessage(), e);
        }
    }

    private String execute(Request request, OkHttpClient client, String operation) {
        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body() != null ? response.body().string() : "";
            if (!response.isSuccessful()) {
                throw new RuntimeException(providerLabel() + " " + operation + "失败: HTTP " + response.code()
                        + (StrUtil.isNotBlank(responseBody) ? " - " + responseBody : ""));
            }
            return responseBody;
        } catch (IOException e) {
            throw new RuntimeException(providerLabel() + " " + operation + "调用异常: " + e.getMessage(), e);
        }
    }

    private OpenAiCompatibleImageProtocolContext buildContext(AiModel model,
                                                              ApiConfig apiConfig,
                                                              String modelCode,
                                                              String prompt,
                                                              int width,
                                                              int height,
                                                              int count,
                                                              List<String> imageUrls,
                                                              JSONObject modelConfig) {
        return new OpenAiCompatibleImageProtocolContext(
                model, apiConfig, modelCode, prompt, width, height, Math.max(count, 1),
                imageUrls, modelConfig, protocolSupport.isAsyncMode(modelConfig));
    }


    private void validateApiConfig(ApiConfig apiConfig) {
        if (apiConfig == null || StrUtil.isBlank(apiConfig.getApiKey())) {
            throw new BusinessException(providerLabel() + " 图片模型缺少 apiKey 配置");
        }
    }

    protected String defaultModelCode() {
        return OpenAiCompatibleImageProtocolSupport.DEFAULT_IMAGE_MODEL;
    }

    protected String providerLabel() {
        return getName();
    }


    private static OkHttpClient defaultHttpClient() {
        return GenerationHttpClients.providerRequestBuilder()
                .connectTimeout(1, TimeUnit.MINUTES)
                .readTimeout(25, TimeUnit.MINUTES)
                .writeTimeout(30, TimeUnit.SECONDS)
                .build();
    }
}
