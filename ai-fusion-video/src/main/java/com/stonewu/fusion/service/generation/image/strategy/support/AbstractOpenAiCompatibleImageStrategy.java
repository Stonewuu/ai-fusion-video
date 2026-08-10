package com.stonewu.fusion.service.generation.image.strategy.support;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategy;
import lombok.extern.slf4j.Slf4j;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

import java.io.IOException;
import java.util.ArrayList;
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
    private final OkHttpClient okHttpClient;

    protected AbstractOpenAiCompatibleImageStrategy(ImageGenerationService imageGenerationService,
                                                     AiModelService aiModelService,
                                                     OpenAiCompatibleImageProtocolSupport protocolSupport,
                                                     OpenAiCompatibleImageProtocolAdapter protocolAdapter) {
        this.imageGenerationService = imageGenerationService;
        this.aiModelService = aiModelService;
        this.protocolSupport = protocolSupport;
        this.protocolAdapter = protocolAdapter;
        this.okHttpClient = defaultHttpClient();
    }

    @Override
    public List<String> generate(String prompt, String modelCode, int width, int height, int count,
                                 List<String> imageUrls, ApiConfig apiConfig) {
        return generate(prompt, null, modelCode, width, height, count, imageUrls, apiConfig);
    }

    @Override
    public List<String> generate(String prompt, AiModel model, int width, int height, int count,
                                 List<String> imageUrls, ApiConfig apiConfig) {
        return generate(prompt, model, model != null ? model.getCode() : null,
                width, height, count, imageUrls, apiConfig);
    }

    private List<String> generate(String prompt, AiModel model, String modelCode,
                                  int width, int height, int count,
                                  List<String> imageUrls, ApiConfig apiConfig) {
        validateApiConfig(apiConfig);
        String actualModelCode = StrUtil.blankToDefault(modelCode, defaultModelCode());
        JSONObject modelConfig = protocolSupport.resolveModelConfig(actualModelCode, model);
        OpenAiCompatibleImageProtocolContext context = buildContext(
                model, apiConfig, actualModelCode, prompt, width, height, count, imageUrls, modelConfig);
        return generateInternal(context);
    }

    @Override
    public String submit(ImageTask task, ApiConfig apiConfig) {
        validateApiConfig(apiConfig);
        AiModel model = resolveModel(task);
        String modelCode = model != null && StrUtil.isNotBlank(model.getCode())
                ? model.getCode() : defaultModelCode();
        JSONObject modelConfig = protocolSupport.resolveModelConfig(modelCode, model);
        int[] size = protocolSupport.resolveConfiguredSize(task, modelConfig);
        int count = task.getCount() != null && task.getCount() > 0 ? task.getCount() : 1;
        OpenAiCompatibleImageProtocolContext context = buildContext(
                model, apiConfig, modelCode, task.getPrompt(), size[0], size[1], count,
                parseRefImageUrls(task.getRefImageUrls()), modelConfig);

        if (context.asyncMode()) {
            String platformTaskId = submitAsyncGeneration(context);
            log.info("[{} Image] 异步任务已提交: taskId={}, platformTaskId={}, model={}",
                    providerLabel(), task.getTaskId(), platformTaskId, modelCode);
            return platformTaskId;
        }

        List<String> urls = generateSync(context);
        updateImageTaskResults(task, urls);
        log.info("[{} Image] 任务完成: taskId={}, model={}, imageCount={}",
                providerLabel(), task.getTaskId(), modelCode, urls.size());
        return task.getTaskId();
    }

    @Override
    public void poll(String platformTaskId, ImageTask task, ApiConfig apiConfig) {
        validateApiConfig(apiConfig);
        AiModel model = resolveModel(task);
        String modelCode = model != null && StrUtil.isNotBlank(model.getCode())
                ? model.getCode() : defaultModelCode();
        JSONObject modelConfig = protocolSupport.resolveModelConfig(modelCode, model);
        if (!protocolSupport.isAsyncMode(modelConfig) || StrUtil.isBlank(platformTaskId)) {
            return;
        }

        List<String> urls = waitForAsyncTask(platformTaskId, apiConfig, modelConfig);
        updateImageTaskResults(task, urls);
        log.info("[{} Image] 异步任务完成: taskId={}, platformTaskId={}, imageCount={}",
                providerLabel(), task.getTaskId(), platformTaskId, urls.size());
    }

    private List<String> generateInternal(OpenAiCompatibleImageProtocolContext context) {
        if (context.asyncMode()) {
            return waitForAsyncTask(submitAsyncGeneration(context), context.apiConfig(), context.modelConfig());
        }
        return generateSync(context);
    }

    private List<String> generateSync(OpenAiCompatibleImageProtocolContext context) {
        int desiredCount = Math.max(context.count(), 1);
        boolean singleImagePerRequest = protocolAdapter.isSingleImagePerRequest(context);
        int requestAttempts = singleImagePerRequest ? desiredCount : 1;
        List<String> generatedUrls = new ArrayList<>();

        for (int attempt = 0; attempt < requestAttempts && generatedUrls.size() < desiredCount; attempt++) {
            OpenAiCompatibleImageProtocolContext requestContext = context.withCount(
                    singleImagePerRequest ? 1 : desiredCount);
            OpenAiCompatibleImageRequest protocolRequest = protocolAdapter.buildRequest(requestContext);
            log.info("[{} Image] 调用生成 API: model={}, size={}x{}, url={}, request={}/{}",
                    providerLabel(), context.modelCode(), context.width(), context.height(), protocolRequest.url(),
                    attempt + 1, requestAttempts);
            String responseBody = execute(protocolRequest, context.apiConfig(), "图片生成");
            generatedUrls.addAll(protocolAdapter.parseImageUrls(requestContext, responseBody));
        }

        return generatedUrls.size() <= desiredCount
                ? generatedUrls : new ArrayList<>(generatedUrls.subList(0, desiredCount));
    }

    private String submitAsyncGeneration(OpenAiCompatibleImageProtocolContext context) {
        OpenAiCompatibleImageProtocolContext asyncContext = context.asyncMode()
                ? context
                : new OpenAiCompatibleImageProtocolContext(
                context.model(), context.apiConfig(), context.modelCode(), context.prompt(),
                context.width(), context.height(), context.count(), context.imageUrls(),
                context.modelConfig(), true);
        OpenAiCompatibleImageRequest protocolRequest = protocolAdapter.buildRequest(asyncContext);
        log.info("[{} Image] 调用异步提交 API: model={}, size={}x{}, url={}",
                providerLabel(), context.modelCode(), context.width(), context.height(), protocolRequest.url());
        return protocolSupport.parseAsyncTaskId(
                execute(protocolRequest, context.apiConfig(), "异步图片任务提交"));
    }

    private List<String> waitForAsyncTask(String platformTaskId, ApiConfig apiConfig, JSONObject modelConfig) {
        sleepQuietly(protocolSupport.resolveAsyncInitialDelayMillis(modelConfig));
        long pollInterval = protocolSupport.resolveAsyncPollIntervalMillis(modelConfig);
        long deadline = System.currentTimeMillis() + protocolSupport.resolveAsyncTimeoutMillis(modelConfig);

        while (System.currentTimeMillis() <= deadline) {
            OpenAiCompatibleImageAsyncTaskResult result = fetchAsyncTaskResult(
                    platformTaskId, apiConfig, modelConfig);
            if (result.completed()) {
                if (result.urls().isEmpty()) {
                    throw new RuntimeException(providerLabel() + " 异步图片任务已完成但未返回图片 URL");
                }
                return result.urls();
            }
            if (result.failed()) {
                throw new RuntimeException(providerLabel() + " 异步图片任务失败: "
                        + StrUtil.blankToDefault(result.errorMessage(), "未知错误"));
            }
            sleepQuietly(pollInterval);
        }
        throw new RuntimeException(providerLabel() + " 异步图片任务轮询超时: taskId=" + platformTaskId);
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

    private String execute(OpenAiCompatibleImageRequest protocolRequest,
                           ApiConfig apiConfig,
                           String operation) {
        Request request = new Request.Builder()
                .url(protocolRequest.url())
                .addHeader("Authorization", "Bearer " + apiConfig.getApiKey())
                .post(protocolRequest.body())
                .build();
        OkHttpClient client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
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

    private void updateImageTaskResults(ImageTask task, List<String> urls) {
        List<ImageItem> items = imageGenerationService.listItems(task.getId());
        for (int i = 0; i < urls.size() && i < items.size(); i++) {
            ImageItem item = items.get(i);
            item.setImageUrl(urls.get(i));
            item.setStatus(1);
            imageGenerationService.updateItem(item);
        }
        task.setSuccessCount(Math.min(urls.size(), items.size()));
        imageGenerationService.update(task);
    }

    private AiModel resolveModel(ImageTask task) {
        if (task != null && task.getModelId() != null) {
            AiModel model = aiModelService.getById(task.getModelId());
            if (model != null && StrUtil.isNotBlank(model.getCode())) {
                return model;
            }
        }
        return null;
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

    private void sleepQuietly(long millis) {
        if (millis <= 0) return;
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(providerLabel() + " 异步图片任务轮询被中断", e);
        }
    }

    private static OkHttpClient defaultHttpClient() {
        return new OkHttpClient.Builder()
                .connectTimeout(1, TimeUnit.MINUTES)
                .readTimeout(25, TimeUnit.MINUTES)
                .writeTimeout(30, TimeUnit.SECONDS)
                .build();
    }
}
