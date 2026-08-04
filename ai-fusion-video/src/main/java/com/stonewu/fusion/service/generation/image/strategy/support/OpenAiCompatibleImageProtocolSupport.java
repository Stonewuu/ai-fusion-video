package com.stonewu.fusion.service.generation.image.strategy.support;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.storage.MediaStorageService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

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
 * Shared mechanics for OpenAI-compatible image protocols.
 *
 * <p>This class owns HTTP shape construction, response normalization, media
 * loading, URL resolution and model-size mapping.  It intentionally contains
 * no task orchestration or persistence logic; those concerns stay in the
 * strategy and protocol adapters.</p>
 */
@Component
public class OpenAiCompatibleImageProtocolSupport {

    public static final String DEFAULT_BASE_URL = "https://api.openai.com";
    public static final String DEFAULT_IMAGE_MODEL = "gpt-image-1";
    private static final String DEFAULT_LOCAL_MEDIA_BASE_PATH = "./data/media";
    private static final int RESPONSE_PREVIEW_LENGTH = 240;
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final MediaType JSON_MEDIA_TYPE = MediaType.get("application/json");

    private final ModelPresetService modelPresetService;
    private final MediaStorageService mediaStorageService;
    private final StorageConfigService storageConfigService;
    private final PresetArtStyleResourceResolver presetArtStyleResourceResolver;
    private final OkHttpClient okHttpClient;

    @Autowired
    public OpenAiCompatibleImageProtocolSupport(ModelPresetService modelPresetService,
                                                MediaStorageService mediaStorageService,
                                                StorageConfigService storageConfigService,
                                                PresetArtStyleResourceResolver presetArtStyleResourceResolver) {
        this(modelPresetService, mediaStorageService, storageConfigService,
                presetArtStyleResourceResolver, defaultHttpClient());
    }

    /** Convenient constructor for protocol-level tests and embedded callers. */
    public OpenAiCompatibleImageProtocolSupport(ModelPresetService modelPresetService,
                                                MediaStorageService mediaStorageService,
                                                StorageConfigService storageConfigService,
                                                PresetArtStyleResourceResolver presetArtStyleResourceResolver,
                                                OkHttpClient okHttpClient) {
        this.modelPresetService = modelPresetService;
        this.mediaStorageService = mediaStorageService;
        this.storageConfigService = storageConfigService;
        this.presetArtStyleResourceResolver = presetArtStyleResourceResolver;
        this.okHttpClient = okHttpClient == null ? defaultHttpClient() : okHttpClient;
    }

    /** Build a request containing only fields defined by the official OpenAI Images API. */
    public OpenAiCompatibleImageRequest buildOpenAiGenerationsRequest(OpenAiCompatibleImageProtocolContext context) {
        try {
            var root = OBJECT_MAPPER.createObjectNode();
            root.put("prompt", StrUtil.blankToDefault(context.prompt(), ""));
            root.put("model", StrUtil.blankToDefault(context.modelCode(), DEFAULT_IMAGE_MODEL));
            root.put("n", Math.max(context.count(), 1));
            root.put("size", mapSize(context.modelCode(), context.width(), context.height(), context.modelConfig()));
            appendOptionalString(root, "quality", getString(context.modelConfig(), "quality", "imageQuality"));
            appendOptionalString(root, "background", getString(context.modelConfig(), "background"));
            appendOptionalString(root, "moderation", getString(context.modelConfig(), "moderation"));
            appendOptionalString(root, "output_format", getString(context.modelConfig(), "outputFormat", "output_format"));
            appendOptionalInteger(root, "output_compression",
                    getInteger(context.modelConfig(), "outputCompression", "output_compression"));
            appendOptionalString(root, "response_format",
                    getString(context.modelConfig(), "responseFormat", "response_format"));
            appendOptionalString(root, "style", getString(context.modelConfig(), "style"));
            appendOptionalString(root, "user", getString(context.modelConfig(), "user", "endUser", "end_user"));
            return jsonRequest(resolveImagesGenerateUrl(context.apiConfig()), root);
        } catch (Exception e) {
            throw new RuntimeException("构建 OpenAI 图片请求失败: " + e.getMessage(), e);
        }
    }

    /** Build an OpenAI-shaped request with NewAPI-compatible extension fields. */
    public OpenAiCompatibleImageRequest buildNewApiGenerationsRequest(OpenAiCompatibleImageProtocolContext context) {
        try {
            var root = OBJECT_MAPPER.createObjectNode();
            root.put("prompt", StrUtil.blankToDefault(context.prompt(), ""));
            root.put("model", StrUtil.blankToDefault(context.modelCode(), DEFAULT_IMAGE_MODEL));
            root.put("n", Math.max(context.count(), 1));
            root.put("size", mapSize(context.modelCode(), context.width(), context.height(), context.modelConfig()));
            if (context.hasReferenceImages()) {
                var imageUrlArray = root.putArray("image_urls");
                context.imageUrls().stream()
                        .filter(StrUtil::isNotBlank)
                        .map(String::trim)
                        .forEach(imageUrlArray::add);
            }
            appendOptionalString(root, "quality", getString(context.modelConfig(), "quality", "imageQuality"));
            appendOptionalString(root, "resolution", getString(context.modelConfig(), "resolution", "defaultResolution"));
            appendOptionalString(root, "background", getString(context.modelConfig(), "background"));
            appendOptionalString(root, "moderation", getString(context.modelConfig(), "moderation"));
            appendOptionalString(root, "output_format", getString(context.modelConfig(), "outputFormat", "output_format"));
            appendOptionalInteger(root, "output_compression",
                    getInteger(context.modelConfig(), "outputCompression", "output_compression"));
            appendOptionalString(root, "response_format",
                    getString(context.modelConfig(), "responseFormat", "response_format"));
            appendOptionalString(root, "mask_url", getString(context.modelConfig(), "maskUrl", "mask_url"));
            appendOptionalString(root, "style", getString(context.modelConfig(), "style"));
            return jsonRequest(resolveImagesGenerateUrl(context.apiConfig()), root);
        } catch (Exception e) {
            throw new RuntimeException("构建 NewAPI 图片请求失败: " + e.getMessage(), e);
        }
    }

    /** Build the shared Agnes Image 2.0/2.1 generations protocol. */
    public OpenAiCompatibleImageRequest buildAgnesGenerationsRequest(OpenAiCompatibleImageProtocolContext context) {
        try {
            var root = OBJECT_MAPPER.createObjectNode();
            root.put("model", StrUtil.blankToDefault(context.modelCode(), DEFAULT_IMAGE_MODEL));
            root.put("prompt", StrUtil.blankToDefault(context.prompt(), ""));

            AgnesImageSize requestSize = resolveAgnesImageSize(
                    context.modelCode(), context.width(), context.height(), context.modelConfig());
            root.put("size", requestSize.size());
            if (StrUtil.isNotBlank(requestSize.ratio())) {
                root.put("ratio", requestSize.ratio());
            }

            var extraBody = OBJECT_MAPPER.createObjectNode();
            if (context.hasReferenceImages()) {
                var imageArray = extraBody.putArray("image");
                context.imageUrls().stream()
                        .filter(StrUtil::isNotBlank)
                        .map(String::trim)
                        .map(imageUrl -> resolveAgnesImageInput(imageUrl, context))
                        .forEach(imageArray::add);
            }

            String responseFormat = StrUtil.blankToDefault(
                    getString(context.modelConfig(), "agnesResponseFormat", "responseFormat", "response_format"),
                    "url");
            boolean returnBase64 = Boolean.TRUE.equals(getBoolean(context.modelConfig(),
                    "returnBase64", "return_base64")) || "b64_json".equalsIgnoreCase(responseFormat);
            if (returnBase64 && !context.hasReferenceImages()) {
                root.put("return_base64", true);
            } else {
                extraBody.put("response_format", returnBase64 ? "b64_json" : responseFormat.trim());
            }
            if (!extraBody.isEmpty()) {
                root.set("extra_body", extraBody);
            }
            return jsonRequest(resolveImagesGenerateUrl(context.apiConfig()), root);
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException("构建 Agnes 图片请求失败: " + e.getMessage(), e);
        }
    }

    /** Build an official OpenAI image edit request using the model's declared reference-image transports. */
    public OpenAiCompatibleImageRequest buildOpenAiEditsRequest(OpenAiCompatibleImageProtocolContext context) {
        if (supportsReferenceImageUrlInput(context.modelConfig())) {
            return buildOpenAiJsonEditsRequest(context);
        }
        MultipartBody.Builder builder = createEditsRequestBuilder(context);
        appendOptionalFormField(builder, "quality", getString(context.modelConfig(), "quality", "imageQuality"));
        appendOptionalFormField(builder, "background", getString(context.modelConfig(), "background"));
        appendOptionalFormField(builder, "moderation", getString(context.modelConfig(), "moderation"));
        appendOptionalFormField(builder, "output_format", getString(context.modelConfig(), "outputFormat", "output_format"));
        appendOptionalFormField(builder, "output_compression",
                getString(context.modelConfig(), "outputCompression", "output_compression"));
        if (!"gpt-image-2".equalsIgnoreCase(StrUtil.trim(context.modelCode()))) {
            appendOptionalFormField(builder, "input_fidelity",
                    getString(context.modelConfig(), "inputFidelity", "input_fidelity"));
        }
        appendOptionalFormField(builder, "user", getString(context.modelConfig(), "user", "endUser", "end_user"));
        appendReferenceImages(builder, context);
        return new OpenAiCompatibleImageRequest(resolveImagesEditUrl(context.apiConfig()), builder.build());
    }

    /** Build an image edit request with fields commonly accepted by NewAPI-compatible gateways. */
    public OpenAiCompatibleImageRequest buildNewApiEditsRequest(OpenAiCompatibleImageProtocolContext context) {
        if (supportsReferenceImageUrlInput(context.modelConfig())) {
            return buildNewApiJsonEditsRequest(context);
        }
        MultipartBody.Builder builder = createEditsRequestBuilder(context);

        appendOptionalFormField(builder, "quality", getString(context.modelConfig(), "quality", "imageQuality"));
        appendOptionalFormField(builder, "resolution", getString(context.modelConfig(), "resolution", "defaultResolution"));
        appendOptionalFormField(builder, "background", getString(context.modelConfig(), "background"));
        appendOptionalFormField(builder, "moderation", getString(context.modelConfig(), "moderation"));
        appendOptionalFormField(builder, "output_format", getString(context.modelConfig(), "outputFormat", "output_format"));
        appendOptionalFormField(builder, "output_compression",
                getString(context.modelConfig(), "outputCompression", "output_compression"));
        appendOptionalFormField(builder, "style", getString(context.modelConfig(), "style"));

        appendReferenceImages(builder, context);
        return new OpenAiCompatibleImageRequest(resolveImagesEditUrl(context.apiConfig()), builder.build());
    }

    private OpenAiCompatibleImageRequest buildOpenAiJsonEditsRequest(
            OpenAiCompatibleImageProtocolContext context) {
        try {
            ObjectNode root = createJsonEditsRoot(context);
            appendOptionalString(root, "quality", getString(context.modelConfig(), "quality", "imageQuality"));
            appendOptionalString(root, "background", getString(context.modelConfig(), "background"));
            appendOptionalString(root, "moderation", getString(context.modelConfig(), "moderation"));
            appendOptionalString(root, "output_format",
                    getString(context.modelConfig(), "outputFormat", "output_format"));
            appendOptionalInteger(root, "output_compression",
                    getInteger(context.modelConfig(), "outputCompression", "output_compression"));
            if (!"gpt-image-2".equalsIgnoreCase(StrUtil.trim(context.modelCode()))) {
                appendOptionalString(root, "input_fidelity",
                        getString(context.modelConfig(), "inputFidelity", "input_fidelity"));
            }
            appendOptionalString(root, "user", getString(context.modelConfig(), "user", "endUser", "end_user"));
            return jsonRequest(resolveImagesEditUrl(context.apiConfig()), root);
        } catch (IOException e) {
            throw new RuntimeException("构建 OpenAI 图片编辑 JSON 请求失败: " + e.getMessage(), e);
        }
    }

    private OpenAiCompatibleImageRequest buildNewApiJsonEditsRequest(
            OpenAiCompatibleImageProtocolContext context) {
        try {
            ObjectNode root = createJsonEditsRoot(context);
            appendOptionalString(root, "quality", getString(context.modelConfig(), "quality", "imageQuality"));
            appendOptionalString(root, "resolution",
                    getString(context.modelConfig(), "resolution", "defaultResolution"));
            appendOptionalString(root, "background", getString(context.modelConfig(), "background"));
            appendOptionalString(root, "moderation", getString(context.modelConfig(), "moderation"));
            appendOptionalString(root, "output_format",
                    getString(context.modelConfig(), "outputFormat", "output_format"));
            appendOptionalInteger(root, "output_compression",
                    getInteger(context.modelConfig(), "outputCompression", "output_compression"));
            appendOptionalString(root, "style", getString(context.modelConfig(), "style"));
            return jsonRequest(resolveImagesEditUrl(context.apiConfig()), root);
        } catch (IOException e) {
            throw new RuntimeException("构建 NewAPI 图片编辑 JSON 请求失败: " + e.getMessage(), e);
        }
    }

    private ObjectNode createJsonEditsRoot(OpenAiCompatibleImageProtocolContext context) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("model", StrUtil.blankToDefault(context.modelCode(), DEFAULT_IMAGE_MODEL));
        root.put("prompt", StrUtil.blankToDefault(context.prompt(), ""));
        root.put("n", Math.max(context.count(), 1));
        root.put("size", mapSize(context.modelCode(), context.width(), context.height(), context.modelConfig()));
        var images = root.putArray("images");
        context.imageUrls().stream()
                .filter(StrUtil::isNotBlank)
                .map(String::trim)
                .forEach(imageUrl -> images.addObject().put("image_url", imageUrl));
        return root;
    }

    private boolean supportsReferenceImageUrlInput(JSONObject config) {
        return getStringList(config, "referenceImageInputFormats").stream()
                .map(value -> StrUtil.blankToDefault(value, "").trim())
                .anyMatch("url"::equalsIgnoreCase);
    }

    private MultipartBody.Builder createEditsRequestBuilder(OpenAiCompatibleImageProtocolContext context) {
        MultipartBody.Builder builder = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("model", StrUtil.blankToDefault(context.modelCode(), DEFAULT_IMAGE_MODEL))
                .addFormDataPart("prompt", StrUtil.blankToDefault(context.prompt(), ""))
                .addFormDataPart("n", String.valueOf(Math.max(context.count(), 1)))
                .addFormDataPart("size", mapSize(context.modelCode(), context.width(), context.height(), context.modelConfig()));
        return builder;
    }

    private void appendReferenceImages(MultipartBody.Builder builder,
                                       OpenAiCompatibleImageProtocolContext context) {
        for (int i = 0; i < context.imageUrls().size(); i++) {
            String imageUrl = context.imageUrls().get(i);
            try {
                BinaryResource resource = loadBinaryResource(imageUrl, context.apiConfig());
                builder.addFormDataPart("image[]", "reference-" + (i + 1) + "." + resource.extension(),
                        RequestBody.create(resource.bytes(), mediaTypeOrDefault(resource.mimeType())));
            } catch (IOException e) {
                throw new RuntimeException("加载 OpenAI 参考图失败: " + e.getMessage(), e);
            }
        }
    }

    public List<String> parseImageUrls(String responseBody) {
        return parseImageUrls(responseBody, true, "兼容图片");
    }

    /** Parse only response fields documented by the official OpenAI Images API. */
    public List<String> parseOpenAiImageUrls(String responseBody) {
        return parseImageUrls(responseBody, false, "OpenAI");
    }

    public List<String> parseAgnesImageUrls(String responseBody) {
        return parseImageUrls(responseBody, false, "Agnes");
    }

    private List<String> parseImageUrls(String responseBody,
                                        boolean allowCompatibleAliases,
                                        String providerLabel) {
        try {
            JsonNode root = OBJECT_MAPPER.readTree(responseBody);
            String errorMessage = extractErrorMessage(root);
            if (StrUtil.isNotBlank(errorMessage)) {
                throw new RuntimeException(providerLabel + " 图片生成失败: " + errorMessage);
            }
            JsonNode data = root.path("data");
            if (!data.isArray() || data.isEmpty()) {
                throw new RuntimeException(providerLabel + " 返回空结果，响应预览: " + previewResponse(responseBody));
            }
            List<String> urls = new ArrayList<>();
            for (JsonNode item : data) {
                String url = allowCompatibleAliases ? firstText(item, "url", "image_url") : textValue(item, "url");
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
                throw new RuntimeException(providerLabel + " 图片响应中未找到 url 或 b64_json，响应预览: "
                        + previewResponse(responseBody));
            }
            return urls;
        } catch (IOException e) {
            throw new RuntimeException("解析 " + providerLabel + " 图片响应失败: " + e.getMessage(), e);
        }
    }

    public String parseAsyncTaskId(String responseBody) {
        try {
            JsonNode root = OBJECT_MAPPER.readTree(responseBody);
            String errorMessage = extractErrorMessage(root);
            if (StrUtil.isNotBlank(errorMessage)) {
                throw new RuntimeException("NewAPI 异步图片任务提交失败: " + errorMessage);
            }
            JsonNode data = root.path("data");
            JsonNode taskNode = data.isArray() && !data.isEmpty() ? data.get(0) : data;
            String taskId = firstText(taskNode, "task_id", "taskId", "id");
            if (StrUtil.isBlank(taskId)) {
                taskId = firstText(root, "task_id", "taskId", "id");
            }
            if (StrUtil.isBlank(taskId)) {
                throw new RuntimeException("NewAPI 异步图片任务响应中未找到 task_id，响应预览: "
                        + previewResponse(responseBody));
            }
            return taskId;
        } catch (IOException e) {
            throw new RuntimeException("解析 NewAPI 异步图片任务响应失败: " + e.getMessage(), e);
        }
    }

    public OpenAiCompatibleImageAsyncTaskResult parseAsyncTaskResult(String responseBody) {
        try {
            JsonNode root = OBJECT_MAPPER.readTree(responseBody);
            String errorMessage = extractErrorMessage(root);
            if (StrUtil.isNotBlank(errorMessage)) {
                return new OpenAiCompatibleImageAsyncTaskResult(null, true, false, List.of(), errorMessage);
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
                return new OpenAiCompatibleImageAsyncTaskResult(status, false, true,
                        extractAsyncImageUrls(root), null);
            }
            if (isAsyncFailureStatus(normalizedStatus)) {
                return new OpenAiCompatibleImageAsyncTaskResult(status, true, false, List.of(),
                        extractAsyncErrorMessage(root, taskNode));
            }
            return new OpenAiCompatibleImageAsyncTaskResult(status, false, false, List.of(), null);
        } catch (IOException e) {
            throw new RuntimeException("解析 NewAPI 异步图片任务查询响应失败: " + e.getMessage(), e);
        }
    }

    public String resolveImagesGenerateUrl(ApiConfig apiConfig) {
        String baseUrl = normalizeBaseUrl(apiConfig);
        if (endsWithIgnoreCase(baseUrl, "/images/generations")) {
            return baseUrl;
        }
        boolean appendV1 = shouldAutoAppendV1Path(apiConfig);
        if (endsWithIgnoreCase(baseUrl, "/v1")) {
            return baseUrl + "/images/generations";
        }
        return baseUrl + (appendV1 ? "/v1/images/generations" : "/images/generations");
    }

    public String resolveAsyncTaskUrl(ApiConfig apiConfig, JSONObject modelConfig, String taskId) {
        String configuredPath = getString(modelConfig, "asyncTaskStatusPath", "asyncTaskPath", "taskStatusPath");
        if (StrUtil.isNotBlank(configuredPath)) {
            String path = configuredPath.trim()
                    .replace("{task_id}", taskId)
                    .replace("{taskId}", taskId)
                    .replace("{id}", taskId);
            if (isHttpUrl(path)) {
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

    public JSONObject resolveModelConfig(String modelCode, AiModel model) {
        JSONObject merged = new JSONObject();
        String capabilityPresetCode = model != null ? model.getCapabilityPresetCode() : null;
        if (modelPresetService != null && StrUtil.isNotBlank(capabilityPresetCode)) {
            mergeConfig(merged, parseConfig(
                    modelPresetService.getPresetConfig(capabilityPresetCode), capabilityPresetCode));
        }
        if (model != null) {
            mergeConfig(merged, parseConfig(model.getConfig(), model.getCode()));
        }
        return merged.isEmpty() ? null : merged;
    }

    public boolean isAsyncMode(JSONObject modelConfig) {
        return Boolean.TRUE.equals(getBoolean(modelConfig,
                "asyncMode", "useAsyncMode", "asyncTaskMode", "enableAsyncTask", "asyncEnabled"));
    }

    public int[] resolveConfiguredSize(ImageTask task, JSONObject modelConfig) {
        int width = task != null && task.getWidth() != null && task.getWidth() > 0 ? task.getWidth() : 0;
        int height = task != null && task.getHeight() != null && task.getHeight() > 0 ? task.getHeight() : 0;
        int[] requestedSize = resolveTaskRequestedSize(task, modelConfig);
        if (requestedSize != null) {
            if (width <= 0) width = requestedSize[0];
            if (height <= 0) height = requestedSize[1];
        }
        if (width <= 0) width = valueOrZero(getInteger(modelConfig, "defaultWidth", "width"));
        if (height <= 0) height = valueOrZero(getInteger(modelConfig, "defaultHeight", "height"));
        if (width <= 0 || height <= 0) {
            int[] fallback = parseSizeText(resolveFallbackSize(modelConfig));
            if (width <= 0) width = fallback[0];
            if (height <= 0) height = fallback[1];
        }
        return new int[]{width > 0 ? width : 1024, height > 0 ? height : 1024};
    }

    public long resolveAsyncInitialDelayMillis(JSONObject config) {
        return secondsToMillis(getIntegerOrDefault(config, 10,
                "asyncTaskInitialDelaySeconds", "asyncInitialDelaySeconds", "taskInitialDelaySeconds"));
    }

    public long resolveAsyncPollIntervalMillis(JSONObject config) {
        return Math.max(100L, secondsToMillis(getIntegerOrDefault(config, 5,
                "asyncTaskPollIntervalSeconds", "asyncPollIntervalSeconds", "taskPollIntervalSeconds")));
    }

    public long resolveAsyncTimeoutMillis(JSONObject config) {
        return secondsToMillis(getIntegerOrDefault(config, 3600,
                "asyncTaskTimeoutSeconds", "asyncTimeoutSeconds", "taskTimeoutSeconds"));
    }

    public Boolean getBoolean(JSONObject config, String... keys) {
        if (config == null) return null;
        for (String key : keys) {
            if (!config.containsKey(key)) continue;
            Object value = config.get(key);
            if (value instanceof Boolean bool) return bool;
            if (value != null) {
                String text = value.toString().trim();
                if ("true".equalsIgnoreCase(text) || "1".equals(text) || "yes".equalsIgnoreCase(text)) return true;
                if ("false".equalsIgnoreCase(text) || "0".equals(text) || "no".equalsIgnoreCase(text)) return false;
            }
        }
        return null;
    }

    public String getString(JSONObject config, String... keys) {
        if (config == null) return null;
        for (String key : keys) {
            Object value = config.get(key);
            if (value != null && StrUtil.isNotBlank(value.toString())) return value.toString().trim();
        }
        return null;
    }

    private OpenAiCompatibleImageRequest jsonRequest(String url, ObjectNode root)
            throws IOException {
        return new OpenAiCompatibleImageRequest(url,
                RequestBody.create(OBJECT_MAPPER.writeValueAsString(root), JSON_MEDIA_TYPE));
    }

    private AgnesImageSize resolveAgnesImageSize(String modelCode, int width, int height, JSONObject config) {
        boolean tierMode = hasTierResolution(config);
        List<ConfiguredImageSize> sizes = collectConfiguredImageSizes(config);
        ConfiguredImageSize selected = null;
        for (ConfiguredImageSize candidate : sizes) {
            if (candidate.width() == width && candidate.height() == height) {
                selected = candidate;
                break;
            }
        }
        if (selected == null) {
            double best = Double.MAX_VALUE;
            for (ConfiguredImageSize candidate : sizes) {
                double distance = imageSizeDistance(width, height, candidate.width(), candidate.height());
                if (distance < best) {
                    selected = candidate;
                    best = distance;
                }
            }
        }
        if (selected != null) {
            if (tierMode && isTier(selected.tier())) {
                return new AgnesImageSize(selected.tier(), selected.ratio());
            }
            return new AgnesImageSize(selected.width() + "x" + selected.height(), null);
        }
        String exact = mapSize(modelCode, width, height, config);
        return new AgnesImageSize(exact, tierMode ? deriveClosestAspectRatio(width, height, config) : null);
    }

    private boolean hasTierResolution(JSONObject config) {
        for (String resolution : getStringList(config, "supportedResolutions")) {
            if (isTier(resolution)) return true;
        }
        JSONObject supportedSizes = asJsonObject(config == null ? null : config.get("supportedSizes"));
        if (supportedSizes != null) {
            for (String key : supportedSizes.keySet()) {
                if (isTier(key)) return true;
            }
        }
        return false;
    }

    private boolean isTier(String value) {
        return StrUtil.isNotBlank(value) && value.trim().matches("\\d+[kK]");
    }

    private String mapSize(String modelCode, int width, int height, JSONObject config) {
        String requested = width + "x" + height;
        if (isConfiguredSizeSupported(config, requested)
                || (Boolean.TRUE.equals(getBoolean(config, "supportCustomSize", "supportsCustomSize",
                "allowCustomSize", "customSizeEnabled")) && isValidConfiguredCustomSize(width, height, config))) {
            return requested;
        }
        String fallback = resolveFallbackSize(config);
        if (!requested.equalsIgnoreCase(fallback)) {
            // Keep this warning in the support layer so all adapters use the
            // same fallback behavior.
            return fallback;
        }
        return fallback;
    }

    private List<ConfiguredImageSize> collectConfiguredImageSizes(JSONObject config) {
        JSONObject supportedSizes = asJsonObject(config == null ? null : config.get("supportedSizes"));
        if (supportedSizes == null || supportedSizes.isEmpty()) return List.of();
        List<ConfiguredImageSize> result = new ArrayList<>();
        for (String tier : supportedSizes.keySet()) {
            JSONObject ratioSizes = asJsonObject(supportedSizes.get(tier));
            if (ratioSizes == null) continue;
            for (String ratio : ratioSizes.keySet()) {
                String text = ratioSizes.getStr(ratio);
                if (!isPixelSizeText(text)) continue;
                int[] dimensions = parseSizeText(text);
                result.add(new ConfiguredImageSize(tier, ratio, dimensions[0], dimensions[1]));
            }
        }
        return result;
    }

    private int[] resolveTaskRequestedSize(ImageTask task, JSONObject config) {
        if (task == null) return null;
        String resolution = StrUtil.blankToDefault(task.getResolution(), "").trim();
        if (isPixelSizeText(resolution)) return parseSizeText(resolution);
        String tier = StrUtil.isNotBlank(resolution) ? resolution : getString(config, "defaultSizeTier", "defaultResolutionTier");
        if (StrUtil.isBlank(tier)) return null;
        String ratio = firstNonBlank(task.getRatio(), task.getAspectRatio(), getString(config, "defaultAspectRatio"));
        ConfiguredImageSize first = null;
        for (ConfiguredImageSize candidate : collectConfiguredImageSizes(config)) {
            if (!candidate.tier().equalsIgnoreCase(tier.trim())) continue;
            if (first == null) first = candidate;
            if (StrUtil.isNotBlank(ratio) && candidate.ratio().equalsIgnoreCase(ratio.trim())) {
                return new int[]{candidate.width(), candidate.height()};
            }
        }
        return first == null ? null : new int[]{first.width(), first.height()};
    }

    private boolean isConfiguredSizeSupported(JSONObject config, String size) {
        return containsConfiguredSize(config == null ? null : config.get("supportedSizes"), size);
    }

    private boolean containsConfiguredSize(Object value, String size) {
        if (value == null) return false;
        if (value instanceof JSONObject object) {
            for (String key : object.keySet()) if (containsConfiguredSize(object.get(key), size)) return true;
            return false;
        }
        if (value instanceof Map<?, ?> map) {
            for (Object item : map.values()) if (containsConfiguredSize(item, size)) return true;
            return false;
        }
        if (value instanceof Iterable<?> iterable) {
            for (Object item : iterable) if (containsConfiguredSize(item, size)) return true;
            return false;
        }
        return size.equalsIgnoreCase(normalizeSizeText(value.toString()));
    }

    private boolean isValidConfiguredCustomSize(int width, int height, JSONObject config) {
        if (width <= 0 || height <= 0) return false;
        Integer multiple = getInteger(config, "sizeMultiple", "imageSizeMultiple", "customSizeMultiple");
        if (multiple != null && multiple > 1 && (width % multiple != 0 || height % multiple != 0)) return false;
        Integer maxEdge = getInteger(config, "maxEdge", "maxImageEdge", "maxDimension");
        if (maxEdge != null && maxEdge > 0 && Math.max(width, height) > maxEdge) return false;
        Integer minEdge = getInteger(config, "minEdge", "minImageEdge", "minDimension");
        if (minEdge != null && minEdge > 0 && Math.min(width, height) < minEdge) return false;
        Double maxRatio = getDouble(config, "maxAspectRatio", "maxRatio");
        if (maxRatio != null && maxRatio > 0 && (double) Math.max(width, height) / Math.min(width, height) > maxRatio) return false;
        long pixels = (long) width * height;
        Long minPixels = getLong(config, "minPixels");
        Long maxPixels = getLong(config, "maxPixels");
        return (minPixels == null || minPixels <= 0 || pixels >= minPixels)
                && (maxPixels == null || maxPixels <= 0 || pixels <= maxPixels);
    }

    private String resolveFallbackSize(JSONObject config) {
        String defaultSize = getString(config, "defaultSize");
        if (isPixelSizeText(defaultSize)) return normalizeSizeText(defaultSize);
        Integer width = getInteger(config, "defaultWidth", "width");
        Integer height = getInteger(config, "defaultHeight", "height");
        if (width != null && width > 0 && height != null && height > 0) return width + "x" + height;
        String first = findFirstConfiguredSize(config == null ? null : config.get("supportedSizes"));
        return StrUtil.isNotBlank(first) ? first : "1024x1024";
    }

    private String findFirstConfiguredSize(Object value) {
        if (value == null) return null;
        if (value instanceof JSONObject object) {
            for (String key : object.keySet()) {
                String found = findFirstConfiguredSize(object.get(key));
                if (StrUtil.isNotBlank(found)) return found;
            }
            return null;
        }
        if (value instanceof Map<?, ?> map) {
            for (Object item : map.values()) {
                String found = findFirstConfiguredSize(item);
                if (StrUtil.isNotBlank(found)) return found;
            }
            return null;
        }
        if (value instanceof Iterable<?> iterable) {
            for (Object item : iterable) {
                String found = findFirstConfiguredSize(item);
                if (StrUtil.isNotBlank(found)) return found;
            }
            return null;
        }
        String text = normalizeSizeText(value.toString());
        return isPixelSizeText(text) ? text : null;
    }

    private double imageSizeDistance(int requestedWidth, int requestedHeight, int candidateWidth, int candidateHeight) {
        if (requestedWidth <= 0 || requestedHeight <= 0 || candidateWidth <= 0 || candidateHeight <= 0) {
            return Double.MAX_VALUE;
        }
        return Math.abs(Math.log((double) candidateWidth / requestedWidth))
                + Math.abs(Math.log((double) candidateHeight / requestedHeight));
    }

    private String deriveClosestAspectRatio(int width, int height, JSONObject config) {
        if (width <= 0 || height <= 0) return StrUtil.blankToDefault(getString(config, "defaultAspectRatio"), "1:1");
        List<String> ratios = getStringList(config, "supportedAspectRatios");
        if (ratios.isEmpty()) ratios = List.of("1:1", "3:4", "4:3", "16:9", "9:16", "2:3", "3:2", "21:9");
        double requestedRatio = (double) width / height;
        String closest = null;
        double score = Double.MAX_VALUE;
        for (String ratio : ratios) {
            String[] parts = StrUtil.blankToDefault(ratio, "").split(":", 2);
            if (parts.length != 2) continue;
            try {
                double candidate = Double.parseDouble(parts[0]) / Double.parseDouble(parts[1]);
                double current = Math.abs(Math.log(candidate / requestedRatio));
                if (current < score) {
                    score = current;
                    closest = ratio.trim();
                }
            } catch (NumberFormatException | ArithmeticException ignored) {
                // Ignore malformed custom ratio entries.
            }
        }
        return StrUtil.blankToDefault(closest, "1:1");
    }

    private BinaryResource loadBinaryResource(String sourceUrl, ApiConfig apiConfig) throws IOException {
        if (StrUtil.isBlank(sourceUrl)) throw new BusinessException("OpenAI 参考图地址为空");
        String trimmed = sourceUrl.trim();
        if (StrUtil.startWithIgnoreCase(trimmed, "data:")) return parseDataUrl(trimmed);
        if (trimmed.startsWith("/media/")) return loadLocalMedia(trimmed);
        if (presetArtStyleResourceResolver != null && presetArtStyleResourceResolver.isPresetArtStylePath(trimmed)) {
            var resource = presetArtStyleResourceResolver.load(trimmed);
            return new BinaryResource(resource.bytes(), resource.mimeType(), extensionFromMimeType(resource.mimeType()));
        }
        if (trimmed.startsWith("file:")) return loadFile(Paths.get(URI.create(trimmed)));
        if (isHttpUrl(trimmed)) {
            Request request = new Request.Builder().url(trimmed).get()
                    .addHeader("Accept", "image/*,*/*;q=0.8").build();
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
        if (Files.exists(localPath) && Files.isRegularFile(localPath)) return loadFile(localPath);
        throw new BusinessException("参考图地址不可访问: " + trimmed);
    }

    private String resolveAgnesImageInput(String imageUrl,
                                          OpenAiCompatibleImageProtocolContext context) {
        String trimmed = StrUtil.blankToDefault(imageUrl, "").trim();
        if (StrUtil.isBlank(trimmed)) {
            throw new BusinessException("Agnes 参考图地址为空");
        }
        if (StrUtil.startWithIgnoreCase(trimmed, "data:")) {
            return trimmed;
        }
        try {
            BinaryResource resource = loadBinaryResource(trimmed, context.apiConfig());
            return "data:" + resource.mimeType() + ";base64,"
                    + Base64.getEncoder().encodeToString(resource.bytes());
        } catch (IOException e) {
            throw unsupportedReferenceImageInput(context.modelCode(),
                    "转换为 Data URI Base64 失败: " + e.getMessage());
        }
    }

    private BusinessException unsupportedReferenceImageInput(String modelCode, String reason) {
        return new BusinessException("图片模型 " + StrUtil.blankToDefault(modelCode, "Agnes")
                + " 无法使用当前参考图：" + reason + "。");
    }

    private BinaryResource parseDataUrl(String sourceUrl) {
        int commaIndex = sourceUrl.indexOf(',');
        if (commaIndex <= 0) throw new BusinessException("OpenAI 参考图 data URL 格式非法");
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
        StorageConfig config = storageConfigService == null ? null : storageConfigService.getDefaultConfig();
        if (config != null && StrUtil.isNotBlank(config.getBasePath())) {
            candidates.add(Paths.get(config.getBasePath()).resolve(relativePath));
        }
        candidates.add(Paths.get(DEFAULT_LOCAL_MEDIA_BASE_PATH).resolve(relativePath));
        for (Path candidate : candidates) {
            if (Files.exists(candidate) && Files.isRegularFile(candidate)) return loadFile(candidate);
        }
        throw new BusinessException("本地参考图不存在: " + sourceUrl);
    }

    private BinaryResource loadFile(Path path) throws IOException {
        String mimeType = normalizeMimeType(null, path.getFileName().toString());
        return new BinaryResource(Files.readAllBytes(path), mimeType, extensionFromMimeType(mimeType));
    }

    private String storeBase64Image(String base64Payload, String extension) {
        String trimmed = StrUtil.trim(base64Payload);
        String actualExtension = extension;
        if (trimmed.startsWith("data:")) {
            int commaIndex = trimmed.indexOf(',');
            String metadata = commaIndex > 0 ? trimmed.substring(0, commaIndex) : trimmed;
            if (commaIndex > 0) trimmed = trimmed.substring(commaIndex + 1);
            actualExtension = resolveExtensionFromMetadata(metadata, actualExtension);
        }
        try {
            byte[] bytes = Base64.getDecoder().decode(trimmed.getBytes(StandardCharsets.UTF_8));
            if (mediaStorageService == null) throw new BusinessException("图片存储服务未配置");
            return mediaStorageService.storeBytes(bytes, "images", actualExtension);
        } catch (IllegalArgumentException e) {
            throw new RuntimeException("OpenAI 返回的图片 base64 数据无效", e);
        }
    }

    private String resolveImageExtension(JsonNode root, JsonNode item) {
        String outputFormat = firstText(root, "output_format");
        if (StrUtil.isBlank(outputFormat)) outputFormat = firstText(item, "output_format", "mime_type");
        if (StrUtil.isBlank(outputFormat)) return "png";
        String normalized = outputFormat.trim().toLowerCase(Locale.ROOT);
        if (normalized.startsWith("image/")) normalized = normalized.substring("image/".length());
        return switch (normalized) {
            case "png", "jpeg", "jpg", "webp", "gif" -> normalized;
            default -> "png";
        };
    }

    private List<String> extractAsyncImageUrls(JsonNode root) {
        List<String> urls = new ArrayList<>();
        JsonNode data = root.path("data");
        JsonNode taskNode = data.isArray() && !data.isEmpty() ? data.get(0) : data;
        collectAsyncImageUrls(taskNode.path("result").path("images"), urls);
        collectAsyncImageUrls(taskNode.path("images"), urls);
        collectAsyncImageUrls(root.path("result").path("images"), urls);
        collectAsyncImageUrls(root.path("images"), urls);
        if (urls.isEmpty() && data.isArray()) for (JsonNode item : data) collectUrlFields(item, urls);
        return urls;
    }

    private void collectAsyncImageUrls(JsonNode node, List<String> urls) {
        if (node == null || node.isMissingNode() || node.isNull()) return;
        if (node.isArray()) {
            for (JsonNode item : node) collectUrlFields(item, urls);
        } else {
            collectUrlFields(node, urls);
        }
    }

    private void collectUrlFields(JsonNode node, List<String> urls) {
        if (node == null || node.isMissingNode() || node.isNull()) return;
        collectUrlValue(node.path("url"), urls);
        collectUrlValue(node.path("urls"), urls);
        collectUrlValue(node.path("image_url"), urls);
        collectUrlValue(node.path("imageUrl"), urls);
    }

    private void collectUrlValue(JsonNode value, List<String> urls) {
        if (value == null || value.isMissingNode() || value.isNull()) return;
        if (value.isArray()) for (JsonNode item : value) collectUrlValue(item, urls);
        else if (value.isTextual() && StrUtil.isNotBlank(value.asText())) urls.add(value.asText().trim());
    }

    private boolean isAsyncSuccessStatus(String status) {
        return "completed".equals(status) || "succeeded".equals(status) || "success".equals(status) || "done".equals(status);
    }

    private boolean isAsyncFailureStatus(String status) {
        return "failed".equals(status) || "error".equals(status) || "cancelled".equals(status) || "canceled".equals(status);
    }

    private String extractAsyncErrorMessage(JsonNode root, JsonNode taskNode) {
        String message = firstText(taskNode, "error", "error_message", "errorMessage", "message");
        return StrUtil.blankToDefault(message, StrUtil.blankToDefault(extractErrorMessage(root), previewResponse(root.toString())));
    }

    private String extractErrorMessage(JsonNode root) {
        if (root == null) return null;
        JsonNode error = root.path("error");
        if (error.isMissingNode() || error.isNull()) return null;
        String message = textValue(error, "message");
        return StrUtil.isNotBlank(message) ? message : error.isTextual() ? error.asText() : previewResponse(error.toString());
    }

    private String firstText(JsonNode node, String... fields) {
        if (node == null || node.isMissingNode() || node.isNull()) return null;
        for (String field : fields) {
            String value = textValue(node, field);
            if (StrUtil.isNotBlank(value)) return value;
        }
        return null;
    }

    private String textValue(JsonNode node, String field) {
        if (node == null) return null;
        JsonNode value = node.path(field);
        if (value.isMissingNode() || value.isNull()) return null;
        String text = value.asText();
        return StrUtil.isBlank(text) ? null : text;
    }

    private String resolveExtensionFromMetadata(String metadata, String fallback) {
        String normalized = metadata == null ? "" : metadata.toLowerCase(Locale.ROOT);
        if (normalized.contains("image/jpeg")) return "jpeg";
        if (normalized.contains("image/jpg")) return "jpg";
        if (normalized.contains("image/webp")) return "webp";
        if (normalized.contains("image/gif")) return "gif";
        return normalized.contains("image/png") ? "png" : StrUtil.blankToDefault(fallback, "png");
    }

    private String normalizeMimeType(String contentType, String sourceUrl) {
        if (StrUtil.isNotBlank(contentType)) return contentType.split(";", 2)[0].trim();
        String lower = sourceUrl == null ? "" : sourceUrl.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
        if (lower.endsWith(".webp")) return "image/webp";
        if (lower.endsWith(".gif")) return "image/gif";
        return "image/png";
    }

    private String extensionFromMimeType(String mimeType) {
        return switch (normalizeMimeType(mimeType, mimeType).toLowerCase(Locale.ROOT)) {
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

    private String resolveImagesEditUrl(ApiConfig apiConfig) {
        String baseUrl = normalizeBaseUrl(apiConfig);
        if (endsWithIgnoreCase(baseUrl, "/images/edits")) return baseUrl;
        if (endsWithIgnoreCase(baseUrl, "/v1")) return baseUrl + "/images/edits";
        return baseUrl + (shouldAutoAppendV1Path(apiConfig) ? "/v1/images/edits" : "/images/edits");
    }

    private String resolveApiRootUrl(ApiConfig apiConfig) {
        String baseUrl = normalizeBaseUrl(apiConfig);
        if (endsWithIgnoreCase(baseUrl, "/images/generations")) {
            return baseUrl.substring(0, baseUrl.length() - "/images/generations".length());
        }
        if (endsWithIgnoreCase(baseUrl, "/images/edits")) {
            return baseUrl.substring(0, baseUrl.length() - "/images/edits".length());
        }
        return baseUrl;
    }

    private boolean shouldAutoAppendV1Path(ApiConfig apiConfig) {
        if (apiConfig == null) return true;
        String platform = apiConfig.getPlatform();
        if (!"openai_compatible".equalsIgnoreCase(platform) && !"agnes".equalsIgnoreCase(platform)) {
            return true;
        }
        return !Boolean.FALSE.equals(apiConfig.getAutoAppendV1Path());
    }

    private String normalizeBaseUrl(ApiConfig apiConfig) {
        String baseUrl = ApiConfigService.resolveEffectiveApiUrlStatic(apiConfig);
        return StrUtil.blankToDefault(baseUrl, DEFAULT_BASE_URL).trim().replaceAll("/+$", "");
    }

    private boolean endsWithIgnoreCase(String text, String suffix) {
        return text != null && suffix != null && text.toLowerCase(Locale.ROOT).endsWith(suffix.toLowerCase(Locale.ROOT));
    }

    private boolean isHttpUrl(String value) {
        return StrUtil.startWithIgnoreCase(value, "http://") || StrUtil.startWithIgnoreCase(value, "https://");
    }

    private JSONObject parseConfig(String configJson, String modelCode) {
        if (StrUtil.isBlank(configJson)) return null;
        try {
            return JSONUtil.parseObj(configJson);
        } catch (Exception e) {
            return null;
        }
    }

    private void mergeConfig(JSONObject target, JSONObject source) {
        if (target == null || source == null || source.isEmpty()) return;
        for (String key : source.keySet()) target.set(key, source.get(key));
    }

    private JSONObject asJsonObject(Object value) {
        if (value instanceof JSONObject object) return object;
        if (value instanceof Map<?, ?> map) return JSONUtil.parseObj(map);
        return null;
    }

    private List<String> getStringList(JSONObject config, String... keys) {
        if (config == null) return List.of();
        for (String key : keys) {
            if (!config.containsKey(key)) continue;
            Object value = config.get(key);
            if (value instanceof Iterable<?> iterable) {
                List<String> result = new ArrayList<>();
                for (Object item : iterable) if (item != null && StrUtil.isNotBlank(item.toString())) result.add(item.toString().trim());
                if (!result.isEmpty()) return result;
            } else if (value != null && StrUtil.isNotBlank(value.toString())) {
                return List.of(value.toString().trim());
            }
        }
        return List.of();
    }

    private Integer getInteger(JSONObject config, String... keys) {
        if (config == null) return null;
        for (String key : keys) {
            if (!config.containsKey(key)) continue;
            Object value = config.get(key);
            if (value instanceof Number number) return number.intValue();
            if (value != null) {
                try { return Integer.parseInt(value.toString().trim()); }
                catch (NumberFormatException ignored) { return null; }
            }
        }
        return null;
    }

    private Long getLong(JSONObject config, String... keys) {
        if (config == null) return null;
        for (String key : keys) {
            if (!config.containsKey(key)) continue;
            Object value = config.get(key);
            if (value instanceof Number number) return number.longValue();
            if (value != null) {
                try { return Long.parseLong(value.toString().trim()); }
                catch (NumberFormatException ignored) { return null; }
            }
        }
        return null;
    }

    private Double getDouble(JSONObject config, String... keys) {
        if (config == null) return null;
        for (String key : keys) {
            if (!config.containsKey(key)) continue;
            Object value = config.get(key);
            if (value instanceof Number number) return number.doubleValue();
            if (value != null) {
                try { return Double.parseDouble(value.toString().trim()); }
                catch (NumberFormatException ignored) { return null; }
            }
        }
        return null;
    }

    private void appendOptionalString(ObjectNode root, String field, String value) {
        if (StrUtil.isNotBlank(value)) root.put(field, value.trim());
    }

    private void appendOptionalInteger(ObjectNode root, String field, Integer value) {
        if (value != null) root.put(field, value);
    }

    private void appendOptionalFormField(MultipartBody.Builder builder, String field, String value) {
        if (StrUtil.isNotBlank(value)) builder.addFormDataPart(field, value.trim());
    }

    private int[] parseSizeText(String sizeText) {
        String normalized = normalizeSizeText(sizeText);
        if (!isPixelSizeText(normalized)) return new int[]{1024, 1024};
        String[] parts = normalized.split("x", 2);
        try { return new int[]{Integer.parseInt(parts[0]), Integer.parseInt(parts[1])}; }
        catch (NumberFormatException ignored) { return new int[]{1024, 1024}; }
    }

    private String normalizeSizeText(String sizeText) {
        return StrUtil.blankToDefault(sizeText, "").trim().toLowerCase(Locale.ROOT).replace('*', 'x');
    }

    private boolean isPixelSizeText(String sizeText) {
        return StrUtil.isNotBlank(sizeText) && normalizeSizeText(sizeText).matches("\\d+x\\d+");
    }

    private Integer getIntegerOrDefault(JSONObject config, int defaultValue, String... keys) {
        Integer value = getInteger(config, keys);
        return value != null && value >= 0 ? value : defaultValue;
    }

    private long secondsToMillis(Integer seconds) {
        return Math.max(seconds == null ? 0 : seconds, 0) * 1000L;
    }

    private int valueOrZero(Integer value) {
        return value == null ? 0 : value;
    }

    private String firstNonBlank(String... values) {
        for (String value : values) if (StrUtil.isNotBlank(value)) return value.trim();
        return null;
    }

    private String previewResponse(String responseBody) {
        if (StrUtil.isBlank(responseBody)) return "<empty>";
        String normalized = responseBody.replaceAll("\\s+", " ").trim();
        return normalized.length() <= RESPONSE_PREVIEW_LENGTH
                ? normalized : normalized.substring(0, RESPONSE_PREVIEW_LENGTH) + "...";
    }

    private static OkHttpClient defaultHttpClient() {
        return new OkHttpClient.Builder()
                .connectTimeout(1, TimeUnit.MINUTES)
                .readTimeout(25, TimeUnit.MINUTES)
                .writeTimeout(30, TimeUnit.SECONDS)
                .build();
    }

    private record BinaryResource(byte[] bytes, String mimeType, String extension) {
    }

    private record AgnesImageSize(String size, String ratio) {
    }

    private record ConfiguredImageSize(String tier, String ratio, int width, int height) {
    }
}
