package com.stonewu.fusion.service.generation.strategy.impl;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.ImageGenerationService;
import com.stonewu.fusion.service.generation.strategy.ImageGenerationStrategy;
import com.stonewu.fusion.service.storage.MediaStorageService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * OpenAI 兼容图片生成策略
 * <p>
 * 仅使用官方 Images API，支持文生图（/images/generations）和参考图编辑（/images/edits）。
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class OpenAiImageStrategy implements ImageGenerationStrategy {

    private static final String DEFAULT_BASE_URL = "https://api.openai.com";
    private static final String DEFAULT_IMAGE_MODEL = "gpt-image-1";
    private static final String DEFAULT_LOCAL_MEDIA_BASE_PATH = "./data/media";
    private static final int DEFAULT_ASYNC_TASK_INITIAL_DELAY_SECONDS = 10;
    private static final int DEFAULT_ASYNC_TASK_POLL_INTERVAL_SECONDS = 5;
    private static final int DEFAULT_ASYNC_TASK_TIMEOUT_SECONDS = 3600;
    private static final int RESPONSE_PREVIEW_LENGTH = 240;
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final MediaType JSON_MEDIA_TYPE = MediaType.get("application/json");

    private final ImageGenerationService imageGenerationService;
    private final AiModelService aiModelService;
    private final ModelPresetService modelPresetService;
    private final MediaStorageService mediaStorageService;
    private final StorageConfigService storageConfigService;
    private final OkHttpClient okHttpClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build();

    @Override
    public String getName() {
        return "openai_compatible";
    }

    @Override
    public List<String> generate(String prompt, String modelCode, int width, int height, int count,
                                  List<String> imageUrls, ApiConfig apiConfig) {
        if (apiConfig == null || StrUtil.isBlank(apiConfig.getApiKey())) {
            throw new BusinessException("OpenAI 图片模型缺少 apiKey 配置");
        }

        String actualModelCode = StrUtil.blankToDefault(modelCode, DEFAULT_IMAGE_MODEL);
        JSONObject modelConfig = resolveModelConfig(actualModelCode, null);
        return generateInternal(prompt, actualModelCode, width, height, count, imageUrls, apiConfig, modelConfig);
    }

    private List<String> generateInternal(String prompt, String modelCode, int width, int height, int count,
                                          List<String> imageUrls,
                                          ApiConfig apiConfig,
                                          JSONObject modelConfig) {
        String actualModelCode = StrUtil.blankToDefault(modelCode, DEFAULT_IMAGE_MODEL);
        if (isAsyncMode(modelConfig)) {
            return generateViaAsyncGenerations(prompt, actualModelCode, width, height, count, imageUrls,
                    apiConfig, modelConfig);
        }
        if (imageUrls != null && !imageUrls.isEmpty()) {
            return generateViaEdits(prompt, actualModelCode, width, height, count, imageUrls, apiConfig, modelConfig);
        }

        return generateViaGenerations(prompt, actualModelCode, width, height, count, null, apiConfig, modelConfig);
    }

    private List<String> generateViaGenerations(String prompt, String modelCode, int width, int height, int count,
                                                List<String> imageUrls,
                                                ApiConfig apiConfig, JSONObject modelConfig) {
        String requestUrl = resolveImagesGenerateUrl(apiConfig);
        String requestBody = buildGenerationRequestBody(prompt, modelCode, width, height, count, imageUrls, modelConfig);

        log.info("[OpenAI] 调用文生图 API: model={}, prompt={}, size={}x{}, url={}",
                modelCode, prompt, width, height, requestUrl);

        Request request = new Request.Builder()
                .url(requestUrl)
                .addHeader("Authorization", "Bearer " + apiConfig.getApiKey())
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(requestBody, JSON_MEDIA_TYPE))
                .build();

        OkHttpClient client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body() != null ? response.body().string() : "";
            if (!response.isSuccessful()) {
                throw new RuntimeException("OpenAI 图片生成失败: HTTP " + response.code()
                        + (StrUtil.isNotBlank(responseBody) ? " - " + responseBody : ""));
            }
            return parseImageUrls(responseBody);
        } catch (IOException e) {
            throw new RuntimeException("OpenAI 调用异常: " + e.getMessage(), e);
        }
    }

    private List<String> generateViaAsyncGenerations(String prompt, String modelCode, int width, int height, int count,
                                                     List<String> imageUrls,
                                                     ApiConfig apiConfig,
                                                     JSONObject modelConfig) {
        String platformTaskId = submitAsyncGeneration(prompt, modelCode, width, height, count, imageUrls,
                apiConfig, modelConfig);
        return waitForAsyncTask(platformTaskId, apiConfig, modelConfig);
    }

    private String submitAsyncGeneration(String prompt, String modelCode, int width, int height, int count,
                                         List<String> imageUrls,
                                         ApiConfig apiConfig,
                                         JSONObject modelConfig) {
        String requestUrl = resolveImagesGenerateUrl(apiConfig);
        String requestBody = buildGenerationRequestBody(prompt, modelCode, width, height, count, imageUrls, modelConfig);

        log.info("[OpenAI] 调用异步生图提交 API: model={}, prompt={}, size={}x{}, url={}",
                modelCode, prompt, width, height, requestUrl);

        Request request = new Request.Builder()
                .url(requestUrl)
                .addHeader("Authorization", "Bearer " + apiConfig.getApiKey())
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(requestBody, JSON_MEDIA_TYPE))
                .build();

        OkHttpClient client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body() != null ? response.body().string() : "";
            if (!response.isSuccessful()) {
                throw new RuntimeException("OpenAI 异步图片任务提交失败: HTTP " + response.code()
                        + (StrUtil.isNotBlank(responseBody) ? " - " + responseBody : ""));
            }
            return parseAsyncTaskId(responseBody);
        } catch (IOException e) {
            throw new RuntimeException("OpenAI 异步图片任务提交异常: " + e.getMessage(), e);
        }
    }

    private List<String> waitForAsyncTask(String platformTaskId, ApiConfig apiConfig, JSONObject modelConfig) {
        sleepQuietly(resolveAsyncInitialDelayMillis(modelConfig));

        long pollIntervalMillis = resolveAsyncPollIntervalMillis(modelConfig);
        long deadline = System.currentTimeMillis() + resolveAsyncTimeoutMillis(modelConfig);

        while (System.currentTimeMillis() <= deadline) {
            AsyncTaskResult result = fetchAsyncTaskResult(platformTaskId, apiConfig, modelConfig);
            if (result.completed()) {
                if (result.urls().isEmpty()) {
                    throw new RuntimeException("OpenAI 异步图片任务已完成但未返回图片 URL");
                }
                return result.urls();
            }
            if (result.failed()) {
                throw new RuntimeException("OpenAI 异步图片任务失败: "
                        + StrUtil.blankToDefault(result.errorMessage(), "未知错误"));
            }
            sleepQuietly(pollIntervalMillis);
        }

        throw new RuntimeException("OpenAI 异步图片任务轮询超时: taskId=" + platformTaskId);
    }

    private AsyncTaskResult fetchAsyncTaskResult(String platformTaskId, ApiConfig apiConfig, JSONObject modelConfig) {
        String requestUrl = resolveAsyncTaskUrl(apiConfig, modelConfig, platformTaskId);
        Request request = new Request.Builder()
                .url(requestUrl)
                .addHeader("Authorization", "Bearer " + apiConfig.getApiKey())
                .get()
                .build();

        OkHttpClient client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body() != null ? response.body().string() : "";
            if (!response.isSuccessful()) {
                throw new RuntimeException("OpenAI 异步图片任务查询失败: HTTP " + response.code()
                        + (StrUtil.isNotBlank(responseBody) ? " - " + responseBody : ""));
            }
            return parseAsyncTaskResult(responseBody);
        } catch (IOException e) {
            throw new RuntimeException("OpenAI 异步图片任务查询异常: " + e.getMessage(), e);
        }
    }

    @Override
    public String submit(ImageTask task, ApiConfig apiConfig) {
        AiModel model = resolveModel(task);
        String modelCode = (model != null && StrUtil.isNotBlank(model.getCode())) ? model.getCode() : "dall-e-3";
        JSONObject modelConfig = resolveModelConfig(modelCode, model);
        int[] size = resolveConfiguredSize(task, modelConfig);
        int count = (task.getCount() != null && task.getCount() > 0) ? task.getCount() : 1;

        // 解析参考图（图生图场景）
        List<String> imageUrls = parseRefImageUrls(task.getRefImageUrls());

        if (isAsyncMode(modelConfig)) {
            String platformTaskId = submitAsyncGeneration(task.getPrompt(), modelCode, size[0], size[1], count,
                    imageUrls, apiConfig, modelConfig);
            log.info("[OpenAI] 异步生图已提交: taskId={}, platformTaskId={}", task.getTaskId(), platformTaskId);
            return platformTaskId;
        }

        List<String> urls = generateInternal(task.getPrompt(), modelCode, size[0], size[1], count, imageUrls,
            apiConfig, modelConfig);

        updateImageTaskResults(task, urls);

        log.info("[OpenAI] 文生图完成: taskId={}, imageCount={}", task.getTaskId(), urls.size());
        return task.getTaskId();
    }

    @Override
    public void poll(String platformTaskId, ImageTask task, ApiConfig apiConfig) {
        AiModel model = resolveModel(task);
        String modelCode = (model != null && StrUtil.isNotBlank(model.getCode())) ? model.getCode() : DEFAULT_IMAGE_MODEL;
        JSONObject modelConfig = resolveModelConfig(modelCode, model);
        if (!isAsyncMode(modelConfig) || StrUtil.isBlank(platformTaskId)) {
            return;
        }

        List<String> urls = waitForAsyncTask(platformTaskId, apiConfig, modelConfig);
        updateImageTaskResults(task, urls);
        log.info("[OpenAI] 异步生图完成: taskId={}, platformTaskId={}, imageCount={}",
                task.getTaskId(), platformTaskId, urls.size());
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
        if (task.getModelId() != null) {
            AiModel model = aiModelService.getById(task.getModelId());
            if (model != null && StrUtil.isNotBlank(model.getCode())) {
                return model;
            }
        }
        return null;
    }

    /**
     * 将像素尺寸映射到 OpenAI 支持的尺寸值。
     * 支持的固定尺寸、自定义尺寸开关和尺寸约束均从模型 config JSON 读取。
     */
    private String mapSize(String modelCode, int width, int height, JSONObject modelConfig) {
        String actualModelCode = StrUtil.blankToDefault(modelCode, DEFAULT_IMAGE_MODEL);
        String sizeStr = width + "x" + height;
        if (isConfiguredSizeSupported(modelConfig, sizeStr)
                || (isCustomSizeEnabled(modelConfig) && isValidConfiguredCustomSize(width, height, modelConfig))) {
            return sizeStr;
        }

        String fallback = resolveFallbackSize(modelConfig);
        if (!sizeStr.equalsIgnoreCase(fallback)) {
            log.warn("[OpenAI] 模型 {} 不支持尺寸 {}，回退到 {}", actualModelCode, sizeStr, fallback);
        }
        return fallback;
    }

    private String buildGenerationRequestBody(String prompt, String modelCode, int width, int height, int count,
                                              List<String> imageUrls,
                                              JSONObject modelConfig) {
        try {
            var root = OBJECT_MAPPER.createObjectNode();
            root.put("prompt", prompt);
            root.put("model", StrUtil.blankToDefault(modelCode, DEFAULT_IMAGE_MODEL));
            root.put("n", Math.max(count, 1));
            root.put("size", mapSize(modelCode, width, height, modelConfig));
            if (imageUrls != null && !imageUrls.isEmpty()) {
                var imageUrlArray = root.putArray("image_urls");
                imageUrls.stream()
                        .filter(StrUtil::isNotBlank)
                        .map(String::trim)
                        .forEach(imageUrlArray::add);
            }
            appendOptionalString(root, "quality", getString(modelConfig, "quality", "imageQuality"));
            appendOptionalString(root, "resolution", getString(modelConfig, "resolution", "defaultResolution"));
            appendOptionalString(root, "background", getString(modelConfig, "background"));
            appendOptionalString(root, "moderation", getString(modelConfig, "moderation"));
            appendOptionalString(root, "output_format", getString(modelConfig, "outputFormat", "output_format"));
            appendOptionalInteger(root, "output_compression", getInteger(modelConfig, "outputCompression", "output_compression"));
            appendOptionalString(root, "response_format", getString(modelConfig, "responseFormat", "response_format"));
            appendOptionalString(root, "mask_url", getString(modelConfig, "maskUrl", "mask_url"));
            appendOptionalString(root, "style", getString(modelConfig, "style"));
            return OBJECT_MAPPER.writeValueAsString(root);
        } catch (Exception e) {
            throw new RuntimeException("构建 OpenAI 图片请求失败: " + e.getMessage(), e);
        }
    }

    private List<String> generateViaEdits(String prompt, String modelCode, int width, int height, int count,
                                          List<String> imageUrls,
                                          ApiConfig apiConfig,
                                          JSONObject modelConfig) {
        String requestUrl = resolveImagesEditUrl(apiConfig);
        RequestBody requestBody = buildEditRequestBody(prompt, modelCode, width, height, count, imageUrls,
                apiConfig, modelConfig);
        OkHttpClient client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
        Request request = new Request.Builder()
                .url(requestUrl)
                .addHeader("Authorization", "Bearer " + apiConfig.getApiKey())
                .post(requestBody)
                .build();

        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body() != null ? response.body().string() : "";
            if (!response.isSuccessful()) {
                throw new RuntimeException("OpenAI 图片编辑失败: HTTP " + response.code()
                        + (StrUtil.isNotBlank(responseBody) ? " - " + responseBody : ""));
            }
            return parseImageUrls(responseBody);
        } catch (IOException e) {
            throw new RuntimeException("OpenAI 图片编辑调用异常: " + e.getMessage(), e);
        }
    }

    private RequestBody buildEditRequestBody(String prompt, String modelCode, int width, int height, int count,
                                             List<String> imageUrls,
                                             ApiConfig apiConfig,
                                             JSONObject modelConfig) {
        MultipartBody.Builder builder = new MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("model", StrUtil.blankToDefault(modelCode, DEFAULT_IMAGE_MODEL))
                .addFormDataPart("prompt", prompt)
                .addFormDataPart("n", String.valueOf(Math.max(count, 1)))
            .addFormDataPart("size", mapSize(modelCode, width, height, modelConfig));

        appendOptionalFormField(builder, "quality", getString(modelConfig, "quality", "imageQuality"));
        appendOptionalFormField(builder, "resolution", getString(modelConfig, "resolution", "defaultResolution"));
        appendOptionalFormField(builder, "background", getString(modelConfig, "background"));
        appendOptionalFormField(builder, "moderation", getString(modelConfig, "moderation"));
        appendOptionalFormField(builder, "output_format", getString(modelConfig, "outputFormat", "output_format"));
        appendOptionalFormField(builder, "output_compression", getString(modelConfig, "outputCompression", "output_compression"));
        appendOptionalFormField(builder, "style", getString(modelConfig, "style"));

        for (int i = 0; i < imageUrls.size(); i++) {
            String imageUrl = imageUrls.get(i);
            BinaryResource resource;
            try {
                resource = loadBinaryResource(imageUrl, apiConfig);
            } catch (IOException e) {
                throw new RuntimeException("加载 OpenAI 参考图失败: " + e.getMessage(), e);
            }
            builder.addFormDataPart(
                    "image[]",
                    "reference-" + (i + 1) + "." + resource.extension(),
                    RequestBody.create(resource.bytes(), mediaTypeOrDefault(resource.mimeType()))
            );
        }

        return builder.build();
    }

    private List<String> parseImageUrls(String responseBody) {
        try {
            JsonNode root = OBJECT_MAPPER.readTree(responseBody);
            String errorMessage = extractErrorMessage(root);
            if (StrUtil.isNotBlank(errorMessage)) {
                throw new RuntimeException("OpenAI 图片生成失败: " + errorMessage);
            }

            JsonNode data = root.path("data");
            if (!data.isArray() || data.isEmpty()) {
                throw new RuntimeException("OpenAI 返回空结果，响应预览: " + previewResponse(responseBody));
            }

            List<String> urls = new ArrayList<>();
            for (JsonNode item : data) {
                String url = textValue(item, "url");
                if (StrUtil.isBlank(url)) {
                    url = textValue(item, "image_url");
                }
                if (StrUtil.isNotBlank(url)) {
                    urls.add(url);
                    continue;
                }

                String b64Json = textValue(item, "b64_json");
                if (StrUtil.isNotBlank(b64Json)) {
                    urls.add(storeBase64Image(b64Json, resolveImageExtension(root, item)));
                }
            }

            if (urls.isEmpty()) {
                throw new RuntimeException("OpenAI 图片响应中未找到 url 或 b64_json，响应预览: "
                        + previewResponse(responseBody));
            }
            return urls;
        } catch (IOException e) {
            throw new RuntimeException("解析 OpenAI 图片响应失败: " + e.getMessage(), e);
        }
    }

    private String parseAsyncTaskId(String responseBody) {
        try {
            JsonNode root = OBJECT_MAPPER.readTree(responseBody);
            String errorMessage = extractErrorMessage(root);
            if (StrUtil.isNotBlank(errorMessage)) {
                throw new RuntimeException("OpenAI 异步图片任务提交失败: " + errorMessage);
            }

            JsonNode data = root.path("data");
            JsonNode taskNode = data.isArray() && !data.isEmpty() ? data.get(0) : data;
            String taskId = firstText(taskNode, "task_id", "taskId", "id");
            if (StrUtil.isBlank(taskId)) {
                taskId = firstText(root, "task_id", "taskId", "id");
            }
            if (StrUtil.isBlank(taskId)) {
                throw new RuntimeException("OpenAI 异步图片任务提交响应中未找到 task_id，响应预览: "
                        + previewResponse(responseBody));
            }
            return taskId;
        } catch (IOException e) {
            throw new RuntimeException("解析 OpenAI 异步图片任务提交响应失败: " + e.getMessage(), e);
        }
    }

    private AsyncTaskResult parseAsyncTaskResult(String responseBody) {
        try {
            JsonNode root = OBJECT_MAPPER.readTree(responseBody);
            String errorMessage = extractErrorMessage(root);
            if (StrUtil.isNotBlank(errorMessage)) {
                return new AsyncTaskResult(null, true, false, List.of(), errorMessage);
            }

            JsonNode data = root.path("data");
            JsonNode taskNode = data.isArray() && !data.isEmpty() ? data.get(0) : data;
            if (taskNode.isMissingNode() || taskNode.isNull()) {
                taskNode = root;
            }

            String status = StrUtil.blankToDefault(firstText(taskNode, "status", "state"),
                    firstText(root, "status", "state"));
            String normalizedStatus = status == null ? "" : status.trim().toLowerCase(Locale.ROOT);
            if (isAsyncSuccessStatus(normalizedStatus)) {
                return new AsyncTaskResult(status, false, true, extractAsyncImageUrls(root), null);
            }
            if (isAsyncFailureStatus(normalizedStatus)) {
                return new AsyncTaskResult(status, true, false, List.of(), extractAsyncErrorMessage(root, taskNode));
            }

            return new AsyncTaskResult(status, false, false, List.of(), null);
        } catch (IOException e) {
            throw new RuntimeException("解析 OpenAI 异步图片任务查询响应失败: " + e.getMessage(), e);
        }
    }

    private List<String> extractAsyncImageUrls(JsonNode root) {
        List<String> urls = new ArrayList<>();
        JsonNode data = root.path("data");
        JsonNode taskNode = data.isArray() && !data.isEmpty() ? data.get(0) : data;
        collectAsyncImageUrls(taskNode.path("result").path("images"), urls);
        collectAsyncImageUrls(taskNode.path("images"), urls);
        collectAsyncImageUrls(root.path("result").path("images"), urls);
        collectAsyncImageUrls(root.path("images"), urls);

        if (urls.isEmpty() && data.isArray()) {
            for (JsonNode item : data) {
                collectUrlFields(item, urls);
            }
        }
        return urls;
    }

    private void collectAsyncImageUrls(JsonNode imagesNode, List<String> urls) {
        if (imagesNode == null || imagesNode.isMissingNode() || imagesNode.isNull()) {
            return;
        }
        if (imagesNode.isArray()) {
            for (JsonNode imageNode : imagesNode) {
                collectUrlFields(imageNode, urls);
            }
            return;
        }
        collectUrlFields(imagesNode, urls);
    }

    private void collectUrlFields(JsonNode node, List<String> urls) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return;
        }
        collectUrlValue(node.path("url"), urls);
        collectUrlValue(node.path("urls"), urls);
        collectUrlValue(node.path("image_url"), urls);
        collectUrlValue(node.path("imageUrl"), urls);
    }

    private void collectUrlValue(JsonNode value, List<String> urls) {
        if (value == null || value.isMissingNode() || value.isNull()) {
            return;
        }
        if (value.isArray()) {
            for (JsonNode item : value) {
                collectUrlValue(item, urls);
            }
            return;
        }
        if (value.isTextual() && StrUtil.isNotBlank(value.asText())) {
            urls.add(value.asText().trim());
        }
    }

    private boolean isAsyncSuccessStatus(String status) {
        return "completed".equals(status) || "succeeded".equals(status)
                || "success".equals(status) || "done".equals(status);
    }

    private boolean isAsyncFailureStatus(String status) {
        return "failed".equals(status) || "error".equals(status)
                || "cancelled".equals(status) || "canceled".equals(status);
    }

    private String extractAsyncErrorMessage(JsonNode root, JsonNode taskNode) {
        String message = firstText(taskNode, "error", "error_message", "errorMessage", "message");
        if (StrUtil.isNotBlank(message)) {
            return message;
        }
        return StrUtil.blankToDefault(extractErrorMessage(root), previewResponse(root.toString()));
    }

    private String storeBase64Image(String base64Payload, String extension) {
        String trimmed = StrUtil.trim(base64Payload);
        String actualExtension = extension;

        if (StrUtil.startWithIgnoreCase(trimmed, "data:")) {
            int commaIndex = trimmed.indexOf(',');
            String metadata = commaIndex > 0 ? trimmed.substring(0, commaIndex) : trimmed;
            if (commaIndex > 0) {
                trimmed = trimmed.substring(commaIndex + 1);
            }
            actualExtension = resolveExtensionFromMetadata(metadata, actualExtension);
        }

        byte[] imageBytes;
        try {
            imageBytes = Base64.getDecoder().decode(trimmed.getBytes(StandardCharsets.UTF_8));
        } catch (IllegalArgumentException e) {
            throw new RuntimeException("OpenAI 返回的图片 base64 数据无效", e);
        }
        return mediaStorageService.storeBytes(imageBytes, "images", actualExtension);
    }

    private String resolveImagesEditUrl(ApiConfig apiConfig) {
        String baseUrl = normalizeBaseUrl(StrUtil.blankToDefault(apiConfig != null ? apiConfig.getApiUrl() : null,
                DEFAULT_BASE_URL));
        if (endsWithIgnoreCase(baseUrl, "/images/edits")) {
            return baseUrl;
        }

        boolean appendV1 = shouldAutoAppendV1Path(apiConfig);
        if (endsWithIgnoreCase(baseUrl, "/v1")) {
            return baseUrl + "/images/edits";
        }
        return baseUrl + (appendV1 ? "/v1/images/edits" : "/images/edits");
    }

    private String resolveImagesGenerateUrl(ApiConfig apiConfig) {
        String baseUrl = normalizeBaseUrl(StrUtil.blankToDefault(apiConfig != null ? apiConfig.getApiUrl() : null,
                DEFAULT_BASE_URL));
        if (endsWithIgnoreCase(baseUrl, "/images/generations")) {
            return baseUrl;
        }

        boolean appendV1 = shouldAutoAppendV1Path(apiConfig);
        if (endsWithIgnoreCase(baseUrl, "/v1")) {
            return baseUrl + "/images/generations";
        }
        return baseUrl + (appendV1 ? "/v1/images/generations" : "/images/generations");
    }

    private String resolveAsyncTaskUrl(ApiConfig apiConfig, JSONObject modelConfig, String taskId) {
        String configuredPath = getString(modelConfig, "asyncTaskStatusPath", "asyncTaskPath", "taskStatusPath");
        if (StrUtil.isNotBlank(configuredPath)) {
            String path = configuredPath.trim()
                    .replace("{task_id}", taskId)
                    .replace("{taskId}", taskId)
                    .replace("{id}", taskId);
            if (StrUtil.startWithIgnoreCase(path, "http://") || StrUtil.startWithIgnoreCase(path, "https://")) {
                return path;
            }
            String rootUrl = resolveApiRootUrl(apiConfig);
            return rootUrl + (path.startsWith("/") ? path : "/" + path);
        }

        String rootUrl = resolveApiRootUrl(apiConfig);
        boolean appendV1 = shouldAutoAppendV1Path(apiConfig);
        if (endsWithIgnoreCase(rootUrl, "/v1")) {
            return rootUrl + "/tasks/" + taskId;
        }
        return rootUrl + (appendV1 ? "/v1/tasks/" : "/tasks/") + taskId;
    }

    private String resolveApiRootUrl(ApiConfig apiConfig) {
        String baseUrl = normalizeBaseUrl(StrUtil.blankToDefault(apiConfig != null ? apiConfig.getApiUrl() : null,
                DEFAULT_BASE_URL));
        if (endsWithIgnoreCase(baseUrl, "/images/generations")) {
            return baseUrl.substring(0, baseUrl.length() - "/images/generations".length());
        }
        if (endsWithIgnoreCase(baseUrl, "/images/edits")) {
            return baseUrl.substring(0, baseUrl.length() - "/images/edits".length());
        }
        return baseUrl;
    }

    private boolean shouldAutoAppendV1Path(ApiConfig apiConfig) {
        if (apiConfig == null) {
            return true;
        }
        if (!"openai_compatible".equalsIgnoreCase(apiConfig.getPlatform())) {
            return true;
        }
        return !Boolean.FALSE.equals(apiConfig.getAutoAppendV1Path());
    }

    private String extractErrorMessage(JsonNode root) {
        JsonNode error = root.path("error");
        if (error.isMissingNode() || error.isNull()) {
            return null;
        }

        String message = textValue(error, "message");
        if (StrUtil.isNotBlank(message)) {
            return message;
        }
        return error.isTextual() ? error.asText() : previewResponse(error.toString());
    }

    private BinaryResource loadBinaryResource(String sourceUrl, ApiConfig apiConfig) throws IOException {
        if (StrUtil.isBlank(sourceUrl)) {
            throw new BusinessException("OpenAI 参考图地址为空");
        }
        String trimmed = sourceUrl.trim();
        if (trimmed.startsWith("data:")) {
            return parseDataUrl(trimmed);
        }
        if (trimmed.startsWith("/media/")) {
            return loadLocalMedia(trimmed);
        }
        if (trimmed.startsWith("file:")) {
            return loadFile(Paths.get(URI.create(trimmed)));
        }
        if (StrUtil.startWithIgnoreCase(trimmed, "http://") || StrUtil.startWithIgnoreCase(trimmed, "https://")) {
            Request request = new Request.Builder()
                    .url(trimmed)
                    .get()
                    .addHeader("Accept", "image/*,*/*;q=0.8")
                    .build();
            OkHttpClient client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
            try (Response response = client.newCall(request).execute()) {
                if (!response.isSuccessful() || response.body() == null) {
                    throw new BusinessException("下载参考图失败: HTTP " + response.code() + " url=" + trimmed);
                }
                String mimeType = normalizeMimeType(response.header("Content-Type"), trimmed);
                return new BinaryResource(response.body().bytes(), mimeType, extensionFromMimeType(mimeType));
            }
        }

        Path localPath = Paths.get(trimmed);
        if (Files.exists(localPath) && Files.isRegularFile(localPath)) {
            return loadFile(localPath);
        }
        throw new BusinessException("参考图地址不可访问: " + trimmed);
    }

    private BinaryResource parseDataUrl(String sourceUrl) {
        int commaIndex = sourceUrl.indexOf(',');
        if (commaIndex <= 0) {
            throw new BusinessException("OpenAI 参考图 data URL 格式非法");
        }

        String metadata = sourceUrl.substring(0, commaIndex);
        String payload = sourceUrl.substring(commaIndex + 1);
        String mimeType = normalizeMimeType(metadata.substring("data:".length()), sourceUrl);
        try {
            byte[] bytes = Base64.getDecoder().decode(payload.getBytes(StandardCharsets.UTF_8));
            return new BinaryResource(bytes, mimeType, extensionFromMimeType(mimeType));
        } catch (IllegalArgumentException e) {
            throw new BusinessException("OpenAI 参考图 data URL base64 非法: " + e.getMessage());
        }
    }

    private BinaryResource loadLocalMedia(String sourceUrl) throws IOException {
        String relativePath = sourceUrl.replaceFirst("^/media/?", "");
        List<Path> candidates = new ArrayList<>();
        StorageConfig config = storageConfigService.getDefaultConfig();
        if (config != null && StrUtil.isNotBlank(config.getBasePath())) {
            candidates.add(Paths.get(config.getBasePath()).resolve(relativePath));
        }
        candidates.add(Paths.get(DEFAULT_LOCAL_MEDIA_BASE_PATH).resolve(relativePath));

        for (Path candidate : candidates) {
            if (candidate != null && Files.exists(candidate) && Files.isRegularFile(candidate)) {
                return loadFile(candidate);
            }
        }
        throw new BusinessException("本地参考图不存在: " + sourceUrl);
    }

    private BinaryResource loadFile(Path path) throws IOException {
        String mimeType = normalizeMimeType(null, path.getFileName().toString());
        return new BinaryResource(Files.readAllBytes(path), mimeType, extensionFromMimeType(mimeType));
    }

    private String resolveImageExtension(JsonNode root, JsonNode item) {
        String outputFormat = textValue(root, "output_format");
        if (StrUtil.isBlank(outputFormat)) {
            outputFormat = textValue(item, "output_format");
        }
        if (StrUtil.isBlank(outputFormat)) {
            outputFormat = textValue(item, "mime_type");
        }
        if (StrUtil.isBlank(outputFormat)) {
            return "png";
        }
        String normalized = outputFormat.trim().toLowerCase(Locale.ROOT);
        if (normalized.startsWith("image/")) {
            normalized = normalized.substring("image/".length());
        }
        return switch (normalized) {
            case "png", "jpeg", "jpg", "webp", "gif" -> normalized;
            default -> "png";
        };
    }

    private JSONObject resolveModelConfig(String modelCode, AiModel model) {
        JSONObject merged = new JSONObject();
        String actualModelCode = StrUtil.blankToDefault(modelCode, model != null ? model.getCode() : null);
        if (modelPresetService != null && StrUtil.isNotBlank(actualModelCode)) {
            mergeConfig(merged, parseConfig(modelPresetService.getPresetConfig(actualModelCode), actualModelCode));
        }
        if (model != null) {
            mergeConfig(merged, parseConfig(model.getConfig(), model.getCode()));
        }
        return merged.isEmpty() ? null : merged;
    }

    private JSONObject parseConfig(String configJson, String modelCode) {
        if (StrUtil.isBlank(configJson)) {
            return null;
        }
        try {
            return JSONUtil.parseObj(configJson);
        } catch (Exception e) {
            log.warn("解析图片模型配置失败，已忽略 OpenAI 图片附加参数: modelCode={}, message={}",
                    modelCode, e.getMessage());
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

    private String getString(JSONObject config, String... keys) {
        if (config == null) {
            return null;
        }
        for (String key : keys) {
            String value = config.getStr(key);
            if (StrUtil.isNotBlank(value)) {
                return value;
            }
        }
        return null;
    }

    private Boolean getBoolean(JSONObject config, String... keys) {
        if (config == null) {
            return null;
        }
        for (String key : keys) {
            if (!config.containsKey(key)) {
                continue;
            }
            Object value = config.get(key);
            if (value instanceof Boolean bool) {
                return bool;
            }
            if (value != null) {
                String text = value.toString().trim();
                if ("true".equalsIgnoreCase(text) || "1".equals(text) || "yes".equalsIgnoreCase(text)) {
                    return true;
                }
                if ("false".equalsIgnoreCase(text) || "0".equals(text) || "no".equalsIgnoreCase(text)) {
                    return false;
                }
            }
        }
        return null;
    }

    private Integer getInteger(JSONObject config, String... keys) {
        if (config == null) {
            return null;
        }
        for (String key : keys) {
            if (!config.containsKey(key)) {
                continue;
            }
            Object value = config.get(key);
            if (value instanceof Number number) {
                return number.intValue();
            }
            if (value != null) {
                try {
                    return Integer.parseInt(value.toString().trim());
                } catch (NumberFormatException ignored) {
                    return null;
                }
            }
        }
        return null;
    }

    private Long getLong(JSONObject config, String... keys) {
        if (config == null) {
            return null;
        }
        for (String key : keys) {
            if (!config.containsKey(key)) {
                continue;
            }
            Object value = config.get(key);
            if (value instanceof Number number) {
                return number.longValue();
            }
            if (value != null) {
                try {
                    return Long.parseLong(value.toString().trim());
                } catch (NumberFormatException ignored) {
                    return null;
                }
            }
        }
        return null;
    }

    private Double getDouble(JSONObject config, String... keys) {
        if (config == null) {
            return null;
        }
        for (String key : keys) {
            if (!config.containsKey(key)) {
                continue;
            }
            Object value = config.get(key);
            if (value instanceof Number number) {
                return number.doubleValue();
            }
            if (value != null) {
                try {
                    return Double.parseDouble(value.toString().trim());
                } catch (NumberFormatException ignored) {
                    return null;
                }
            }
        }
        return null;
    }

    private void appendOptionalString(com.fasterxml.jackson.databind.node.ObjectNode root, String fieldName,
                                      String value) {
        if (StrUtil.isNotBlank(value)) {
            root.put(fieldName, value.trim());
        }
    }

    private void appendOptionalInteger(com.fasterxml.jackson.databind.node.ObjectNode root, String fieldName,
                                       Integer value) {
        if (value != null) {
            root.put(fieldName, value);
        }
    }

    private void appendOptionalFormField(MultipartBody.Builder builder, String fieldName, String value) {
        if (StrUtil.isNotBlank(value)) {
            builder.addFormDataPart(fieldName, value.trim());
        }
    }

    private boolean isAsyncMode(JSONObject modelConfig) {
        return Boolean.TRUE.equals(getBoolean(modelConfig,
                "asyncMode", "useAsyncMode", "asyncTaskMode", "enableAsyncTask", "asyncEnabled"));
    }

    private int[] resolveConfiguredSize(ImageTask task, JSONObject modelConfig) {
        int width = (task.getWidth() != null && task.getWidth() > 0) ? task.getWidth() : 0;
        int height = (task.getHeight() != null && task.getHeight() > 0) ? task.getHeight() : 0;

        if (width <= 0) {
            Integer configuredWidth = getInteger(modelConfig, "defaultWidth", "width");
            width = configuredWidth != null ? configuredWidth : 0;
        }
        if (height <= 0) {
            Integer configuredHeight = getInteger(modelConfig, "defaultHeight", "height");
            height = configuredHeight != null ? configuredHeight : 0;
        }

        if (width <= 0 || height <= 0) {
            int[] fallback = parseSizeText(resolveFallbackSize(modelConfig));
            if (width <= 0) {
                width = fallback[0];
            }
            if (height <= 0) {
                height = fallback[1];
            }
        }

        if (width <= 0) width = 1024;
        if (height <= 0) height = 1024;
        return new int[]{width, height};
    }

    private boolean isConfiguredSizeSupported(JSONObject modelConfig, String sizeStr) {
        if (modelConfig == null || StrUtil.isBlank(sizeStr)) {
            return false;
        }
        return containsConfiguredSize(modelConfig.get("supportedSizes"), sizeStr);
    }

    private boolean containsConfiguredSize(Object value, String sizeStr) {
        if (value == null) {
            return false;
        }
        if (value instanceof JSONObject jsonObject) {
            for (String key : jsonObject.keySet()) {
                if (containsConfiguredSize(jsonObject.get(key), sizeStr)) {
                    return true;
                }
            }
            return false;
        }
        if (value instanceof Map<?, ?> map) {
            for (Object item : map.values()) {
                if (containsConfiguredSize(item, sizeStr)) {
                    return true;
                }
            }
            return false;
        }
        if (value instanceof Iterable<?> iterable) {
            for (Object item : iterable) {
                if (containsConfiguredSize(item, sizeStr)) {
                    return true;
                }
            }
            return false;
        }
        return sizeStr.equalsIgnoreCase(normalizeSizeText(value.toString()));
    }

    private boolean isCustomSizeEnabled(JSONObject modelConfig) {
        return Boolean.TRUE.equals(getBoolean(modelConfig,
                "supportCustomSize", "supportsCustomSize", "allowCustomSize", "customSizeEnabled"));
    }

    private boolean isValidConfiguredCustomSize(int width, int height, JSONObject modelConfig) {
        if (width <= 0 || height <= 0) {
            return false;
        }

        Integer sizeMultiple = getInteger(modelConfig, "sizeMultiple", "imageSizeMultiple", "customSizeMultiple");
        if (sizeMultiple != null && sizeMultiple > 1
                && (width % sizeMultiple != 0 || height % sizeMultiple != 0)) {
            return false;
        }

        Integer maxEdge = getInteger(modelConfig, "maxEdge", "maxImageEdge", "maxDimension");
        if (maxEdge != null && maxEdge > 0 && Math.max(width, height) > maxEdge) {
            return false;
        }

        Integer minEdge = getInteger(modelConfig, "minEdge", "minImageEdge", "minDimension");
        if (minEdge != null && minEdge > 0 && Math.min(width, height) < minEdge) {
            return false;
        }

        Double maxAspectRatio = getDouble(modelConfig, "maxAspectRatio", "maxRatio");
        if (maxAspectRatio != null && maxAspectRatio > 0) {
            int longEdge = Math.max(width, height);
            int shortEdge = Math.min(width, height);
            if (shortEdge <= 0 || (double) longEdge / (double) shortEdge > maxAspectRatio) {
                return false;
            }
        }

        long pixels = (long) width * height;
        Long minPixels = getLong(modelConfig, "minPixels");
        if (minPixels != null && minPixels > 0 && pixels < minPixels) {
            return false;
        }
        Long maxPixels = getLong(modelConfig, "maxPixels");
        return maxPixels == null || maxPixels <= 0 || pixels <= maxPixels;
    }

    private String resolveFallbackSize(JSONObject modelConfig) {
        String defaultSize = getString(modelConfig, "defaultSize");
        if (isPixelSizeText(defaultSize)) {
            return normalizeSizeText(defaultSize);
        }

        Integer defaultWidth = getInteger(modelConfig, "defaultWidth", "width");
        Integer defaultHeight = getInteger(modelConfig, "defaultHeight", "height");
        if (defaultWidth != null && defaultWidth > 0 && defaultHeight != null && defaultHeight > 0) {
            return defaultWidth + "x" + defaultHeight;
        }

        String firstConfiguredSize = findFirstConfiguredSize(modelConfig != null ? modelConfig.get("supportedSizes") : null);
        if (StrUtil.isNotBlank(firstConfiguredSize)) {
            return firstConfiguredSize;
        }
        return "1024x1024";
    }

    private String findFirstConfiguredSize(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof JSONObject jsonObject) {
            for (String key : jsonObject.keySet()) {
                String found = findFirstConfiguredSize(jsonObject.get(key));
                if (StrUtil.isNotBlank(found)) {
                    return found;
                }
            }
            return null;
        }
        if (value instanceof Map<?, ?> map) {
            for (Object item : map.values()) {
                String found = findFirstConfiguredSize(item);
                if (StrUtil.isNotBlank(found)) {
                    return found;
                }
            }
            return null;
        }
        if (value instanceof Iterable<?> iterable) {
            for (Object item : iterable) {
                String found = findFirstConfiguredSize(item);
                if (StrUtil.isNotBlank(found)) {
                    return found;
                }
            }
            return null;
        }
        String text = normalizeSizeText(value.toString());
        return isPixelSizeText(text) ? text : null;
    }

    private int[] parseSizeText(String sizeText) {
        String normalized = normalizeSizeText(sizeText);
        if (!isPixelSizeText(normalized)) {
            return new int[]{1024, 1024};
        }
        String[] parts = normalized.split("x", 2);
        try {
            return new int[]{Integer.parseInt(parts[0]), Integer.parseInt(parts[1])};
        } catch (NumberFormatException ignored) {
            return new int[]{1024, 1024};
        }
    }

    private String normalizeSizeText(String sizeText) {
        return StrUtil.blankToDefault(sizeText, "").trim().toLowerCase(Locale.ROOT).replace('*', 'x');
    }

    private boolean isPixelSizeText(String sizeText) {
        return StrUtil.isNotBlank(sizeText) && normalizeSizeText(sizeText).matches("\\d+x\\d+");
    }

    private long resolveAsyncInitialDelayMillis(JSONObject modelConfig) {
        return secondsToMillis(getIntegerOrDefault(modelConfig, DEFAULT_ASYNC_TASK_INITIAL_DELAY_SECONDS,
                "asyncTaskInitialDelaySeconds", "asyncInitialDelaySeconds", "taskInitialDelaySeconds"));
    }

    private long resolveAsyncPollIntervalMillis(JSONObject modelConfig) {
        return Math.max(100L, secondsToMillis(getIntegerOrDefault(modelConfig, DEFAULT_ASYNC_TASK_POLL_INTERVAL_SECONDS,
                "asyncTaskPollIntervalSeconds", "asyncPollIntervalSeconds", "taskPollIntervalSeconds")));
    }

    private long resolveAsyncTimeoutMillis(JSONObject modelConfig) {
        return secondsToMillis(getIntegerOrDefault(modelConfig, DEFAULT_ASYNC_TASK_TIMEOUT_SECONDS,
                "asyncTaskTimeoutSeconds", "asyncTimeoutSeconds", "taskTimeoutSeconds"));
    }

    private Integer getIntegerOrDefault(JSONObject config, int defaultValue, String... keys) {
        Integer value = getInteger(config, keys);
        return value != null && value >= 0 ? value : defaultValue;
    }

    private long secondsToMillis(Integer seconds) {
        return Math.max(seconds != null ? seconds : 0, 0) * 1000L;
    }

    private void sleepQuietly(long millis) {
        if (millis <= 0) {
            return;
        }
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("OpenAI 异步图片任务轮询被中断", e);
        }
    }

    private String firstText(JsonNode node, String... fieldNames) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        for (String fieldName : fieldNames) {
            String value = textValue(node, fieldName);
            if (StrUtil.isNotBlank(value)) {
                return value;
            }
        }
        return null;
    }

    private String resolveExtensionFromMetadata(String metadata, String fallback) {
        String normalized = metadata == null ? "" : metadata.toLowerCase(Locale.ROOT);
        if (normalized.contains("image/jpeg")) {
            return "jpeg";
        }
        if (normalized.contains("image/jpg")) {
            return "jpg";
        }
        if (normalized.contains("image/webp")) {
            return "webp";
        }
        if (normalized.contains("image/gif")) {
            return "gif";
        }
        if (normalized.contains("image/png")) {
            return "png";
        }
        return StrUtil.blankToDefault(fallback, "png");
    }

    private String normalizeMimeType(String contentType, String sourceUrl) {
        if (StrUtil.isNotBlank(contentType)) {
            return contentType.split(";", 2)[0].trim();
        }
        String lower = sourceUrl == null ? "" : sourceUrl.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
            return "image/jpeg";
        }
        if (lower.endsWith(".webp")) {
            return "image/webp";
        }
        if (lower.endsWith(".gif")) {
            return "image/gif";
        }
        return "image/png";
    }

    private String extensionFromMimeType(String mimeType) {
        String normalized = normalizeMimeType(mimeType, mimeType).toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "image/jpeg" -> "jpg";
            case "image/webp" -> "webp";
            case "image/gif" -> "gif";
            default -> "png";
        };
    }

    private MediaType mediaTypeOrDefault(String mimeType) {
        try {
            return MediaType.get(StrUtil.blankToDefault(mimeType, "image/png"));
        } catch (Exception ignored) {
            return MediaType.get("application/octet-stream");
        }
    }

    private String textValue(JsonNode node, String fieldName) {
        if (node == null) {
            return null;
        }
        JsonNode value = node.path(fieldName);
        if (value.isMissingNode() || value.isNull()) {
            return null;
        }
        String text = value.asText();
        return StrUtil.isBlank(text) ? null : text;
    }

    private String normalizeBaseUrl(String baseUrl) {
        return StrUtil.blankToDefault(baseUrl, DEFAULT_BASE_URL).trim().replaceAll("/+$", "");
    }

    private boolean endsWithIgnoreCase(String text, String suffix) {
        return text != null && suffix != null && text.toLowerCase(Locale.ROOT)
                .endsWith(suffix.toLowerCase(Locale.ROOT));
    }

    private String previewResponse(String responseBody) {
        if (StrUtil.isBlank(responseBody)) {
            return "<empty>";
        }
        String normalized = responseBody.replaceAll("\\s+", " ").trim();
        if (normalized.length() <= RESPONSE_PREVIEW_LENGTH) {
            return normalized;
        }
        return normalized.substring(0, RESPONSE_PREVIEW_LENGTH) + "...";
    }

    private record BinaryResource(byte[] bytes, String mimeType, String extension) {
    }

    private record AsyncTaskResult(String status, boolean failed, boolean completed, List<String> urls,
                                   String errorMessage) {
    }
}
