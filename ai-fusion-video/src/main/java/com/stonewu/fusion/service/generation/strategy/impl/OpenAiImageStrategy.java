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
    private static final int GPT_IMAGE_2_SIZE_MULTIPLE = 16;
    private static final int GPT_IMAGE_2_MAX_EDGE = 3840;
    private static final long GPT_IMAGE_2_MIN_PIXELS = 655_360L;
    private static final long GPT_IMAGE_2_MAX_PIXELS = 8_294_400L;
    private static final int RESPONSE_PREVIEW_LENGTH = 240;
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final MediaType JSON_MEDIA_TYPE = MediaType.get("application/json");

    private final ImageGenerationService imageGenerationService;
    private final AiModelService aiModelService;
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

        return generateInternal(prompt, modelCode, width, height, count, imageUrls, apiConfig, null);
    }

    private List<String> generateInternal(String prompt, String modelCode, int width, int height, int count,
                                          List<String> imageUrls,
                                          ApiConfig apiConfig,
                                          JSONObject modelConfig) {
        String actualModelCode = StrUtil.blankToDefault(modelCode, DEFAULT_IMAGE_MODEL);
        if (imageUrls != null && !imageUrls.isEmpty()) {
            return generateViaEdits(prompt, actualModelCode, width, height, count, imageUrls, apiConfig, modelConfig);
        }

        return generateViaGenerations(prompt, actualModelCode, width, height, count, apiConfig, modelConfig);
    }

    private List<String> generateViaGenerations(String prompt, String modelCode, int width, int height, int count,
                                                ApiConfig apiConfig, JSONObject modelConfig) {
        String requestUrl = resolveImagesGenerateUrl(apiConfig);
        String requestBody = buildGenerationRequestBody(prompt, modelCode, width, height, count, modelConfig);

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

    @Override
    public String submit(ImageTask task, ApiConfig apiConfig) {
        AiModel model = resolveModel(task);
        String modelCode = (model != null && StrUtil.isNotBlank(model.getCode())) ? model.getCode() : "dall-e-3";
        int[] size = resolveDefaultSize(model, task);
        int count = (task.getCount() != null && task.getCount() > 0) ? task.getCount() : 1;
        JSONObject modelConfig = parseModelConfig(model);

        // 解析参考图（图生图场景）
        List<String> imageUrls = parseRefImageUrls(task.getRefImageUrls());

        List<String> urls = generateInternal(task.getPrompt(), modelCode, size[0], size[1], count, imageUrls,
            apiConfig, modelConfig);

        // 更新数据库记录
        List<ImageItem> items = imageGenerationService.listItems(task.getId());
        for (int i = 0; i < urls.size() && i < items.size(); i++) {
            ImageItem item = items.get(i);
            item.setImageUrl(urls.get(i));
            item.setStatus(1);
            imageGenerationService.updateItem(item);
        }

        task.setSuccessCount(Math.min(urls.size(), items.size()));
        imageGenerationService.update(task);

        log.info("[OpenAI] 文生图完成: taskId={}, imageCount={}", task.getTaskId(), urls.size());
        return task.getTaskId();
    }

    @Override
    public void poll(String platformTaskId, ImageTask task, ApiConfig apiConfig) {
        // OpenAI images.generate 是同步 API，submit 中已处理完成，无需轮询
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
     * GPT Image 2 按官方约束支持任意合法分辨率，其余模型使用固定枚举值。
     */
    private String mapSize(String modelCode, int width, int height) {
        String actualModelCode = StrUtil.blankToDefault(modelCode, DEFAULT_IMAGE_MODEL);
        String sizeStr = width + "x" + height;
        if ("gpt-image-2".equalsIgnoreCase(actualModelCode)) {
            if (isValidGptImage2Size(width, height)) {
                return sizeStr;
            }
            log.warn("[OpenAI] GPT Image 2 请求尺寸 {} 不满足官方约束，回退到 1024x1024", sizeStr);
            return "1024x1024";
        }
        return switch (sizeStr) {
            case "256x256", "512x512", "1024x1024", "1024x1792", "1792x1024" -> sizeStr;
            case "1024x1536", "1536x1024" -> sizeStr;
            default -> {
                log.warn("[OpenAI] 模型 {} 不支持尺寸 {}，回退到 1024x1024", actualModelCode, sizeStr);
                yield "1024x1024";
            }
        };
    }

    private boolean isValidGptImage2Size(int width, int height) {
        if (width <= 0 || height <= 0) {
            return false;
        }
        if (width % GPT_IMAGE_2_SIZE_MULTIPLE != 0 || height % GPT_IMAGE_2_SIZE_MULTIPLE != 0) {
            return false;
        }
        int longEdge = Math.max(width, height);
        int shortEdge = Math.min(width, height);
        if (longEdge > GPT_IMAGE_2_MAX_EDGE) {
            return false;
        }
        if ((long) longEdge > (long) shortEdge * 3L) {
            return false;
        }
        long pixels = (long) width * height;
        return pixels >= GPT_IMAGE_2_MIN_PIXELS && pixels <= GPT_IMAGE_2_MAX_PIXELS;
    }

    private String buildGenerationRequestBody(String prompt, String modelCode, int width, int height, int count,
                                              JSONObject modelConfig) {
        try {
            var root = OBJECT_MAPPER.createObjectNode();
            root.put("prompt", prompt);
            root.put("model", StrUtil.blankToDefault(modelCode, DEFAULT_IMAGE_MODEL));
            root.put("n", Math.max(count, 1));
            root.put("size", mapSize(modelCode, width, height));
            appendOptionalString(root, "quality", getString(modelConfig, "quality", "imageQuality"));
            appendOptionalString(root, "background", getString(modelConfig, "background"));
            appendOptionalString(root, "moderation", getString(modelConfig, "moderation"));
            appendOptionalString(root, "output_format", getString(modelConfig, "outputFormat", "output_format"));
            appendOptionalString(root, "response_format", getString(modelConfig, "responseFormat", "response_format"));
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
            .addFormDataPart("size", mapSize(modelCode, width, height));

        appendOptionalFormField(builder, "quality", getString(modelConfig, "quality", "imageQuality"));
        appendOptionalFormField(builder, "background", getString(modelConfig, "background"));
        appendOptionalFormField(builder, "moderation", getString(modelConfig, "moderation"));
        appendOptionalFormField(builder, "output_format", getString(modelConfig, "outputFormat", "output_format"));
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

    private JSONObject parseModelConfig(AiModel model) {
        if (model == null || StrUtil.isBlank(model.getConfig())) {
            return null;
        }
        try {
            return JSONUtil.parseObj(model.getConfig());
        } catch (Exception e) {
            log.warn("解析图片模型配置失败，已忽略 OpenAI 图片附加参数: modelCode={}, message={}",
                    model.getCode(), e.getMessage());
            return null;
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

    private void appendOptionalString(com.fasterxml.jackson.databind.node.ObjectNode root, String fieldName,
                                      String value) {
        if (StrUtil.isNotBlank(value)) {
            root.put(fieldName, value.trim());
        }
    }

    private void appendOptionalFormField(MultipartBody.Builder builder, String fieldName, String value) {
        if (StrUtil.isNotBlank(value)) {
            builder.addFormDataPart(fieldName, value.trim());
        }
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
}
