package com.stonewu.fusion.service.generation.video.strategy.support;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * OpenAI 兼容视频协议公共能力。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class OpenAiCompatibleVideoProtocolSupport {

    private static final String DEFAULT_BASE_URL = "https://api.openai.com";
    private static final String DEFAULT_VIDEO_MODEL = "sora-2";
    private static final String DEFAULT_LOCAL_MEDIA_BASE_PATH = "./data/media";
    private static final MediaType JSON_MEDIA_TYPE = MediaType.get("application/json");
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private final StorageConfigService storageConfigService;
    private final PresetArtStyleResourceResolver presetArtStyleResourceResolver;

    private final OkHttpClient resourceHttpClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(300, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build();

    public RequestBody buildSoraSubmitBody(OpenAiCompatibleVideoProtocolContext context) {
        MultipartBody.Builder builder = new MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("model", resolveModelCode(context.model()));

        String prompt = StrUtil.trim(context.task().getPrompt());
        if (StrUtil.isNotBlank(prompt)) {
            builder.addFormDataPart("prompt", prompt);
        }

        String seconds = resolveSeconds(context.task(), context.modelConfig());
        if (StrUtil.isNotBlank(seconds)) {
            builder.addFormDataPart("seconds", seconds);
        }

        String size = resolveSize(context.task(), context.modelConfig());
        if (StrUtil.isNotBlank(size)) {
            builder.addFormDataPart("size", size);
        }

        String referenceImageUrl = resolveSingleReferenceImageUrl(context.task());
        if (StrUtil.isNotBlank(referenceImageUrl)) {
            BinaryResource resource;
            try {
                resource = loadBinaryResource(referenceImageUrl, context.apiConfig());
            } catch (IOException e) {
                throw new BusinessException("加载 OpenAI 视频参考图失败: " + e.getMessage());
            }
            builder.addFormDataPart(
                    "input_reference",
                    "reference." + resource.extension(),
                    RequestBody.create(resource.bytes(), mediaTypeOrDefault(resource.mimeType()))
            );
        }

        return builder.build();
    }

    public RequestBody buildAgnesSubmitBody(OpenAiCompatibleVideoProtocolContext context) {
        JSONObject body = JSONUtil.createObj();
        String modelCode = resolveModelCode(context.model());
        String prompt = StrUtil.trim(context.task().getPrompt());
        if (StrUtil.isBlank(prompt)) {
            throw new BusinessException("Agnes 视频任务缺少 prompt");
        }

        body.set("model", modelCode);
        body.set("prompt", prompt);

        int[] dimensions = resolveAgnesDimensions(context.task(), context.modelConfig());
        body.set("width", dimensions[0]);
        body.set("height", dimensions[1]);

        Integer frameRate = resolveAgnesFrameRate(context.modelConfig());
        if (frameRate != null) {
            body.set("frame_rate", frameRate);
        }

        Integer numFrames = resolveAgnesNumFrames(context.task(), context.modelConfig(), frameRate);
        if (numFrames != null) {
            body.set("num_frames", numFrames);
        }

        appendOptionalInteger(body, "num_inference_steps",
                getPositiveInteger(context.modelConfig(), "numInferenceSteps", "num_inference_steps"));
        appendOptionalString(body, "negative_prompt",
                getString(context.modelConfig(), "negativePrompt", "negative_prompt"));

        if (context.task().getSeed() != null) {
            body.set("seed", context.task().getSeed());
        }

        List<String> imageInputs = resolveAgnesImageInputs(context.task());
        JSONObject extraBody = asJsonObject(context.modelConfig().get("agnesExtraBody"));
        if (extraBody == null) {
            extraBody = asJsonObject(context.modelConfig().get("extraBody"));
        }
        if (extraBody == null) {
            extraBody = new JSONObject();
        }

        if (imageInputs.size() == 1) {
            body.set("image", imageInputs.get(0));
        } else if (!imageInputs.isEmpty()) {
            extraBody.set("image", imageInputs);
        }

        String mode = resolveAgnesMode(context.task(), context.modelConfig(), imageInputs.size());
        if (StrUtil.isNotBlank(mode)) {
            if (!extraBody.isEmpty() || imageInputs.size() > 1) {
                extraBody.set("mode", mode);
            } else {
                body.set("mode", mode);
            }
        }

        if (!extraBody.isEmpty()) {
            body.set("extra_body", extraBody);
        }

        return RequestBody.create(body.toString(), JSON_MEDIA_TYPE);
    }

    public String resolveOpenAiVideosUrl(ApiConfig apiConfig) {
        String baseUrl = normalizeBaseUrl(StrUtil.blankToDefault(apiConfig != null ? apiConfig.getApiUrl() : null,
                DEFAULT_BASE_URL));
        if (endsWithIgnoreCase(baseUrl, "/videos")) {
            return baseUrl;
        }
        if (endsWithIgnoreCase(baseUrl, "/v1")) {
            return baseUrl + "/videos";
        }
        boolean appendV1 = shouldAutoAppendV1Path(apiConfig);
        return baseUrl + (appendV1 ? "/v1/videos" : "/videos");
    }

    public String resolveFixedV1VideosUrl(ApiConfig apiConfig) {
        return resolveApiRoot(apiConfig) + "/v1/videos";
    }

    public String resolveApiRoot(ApiConfig apiConfig) {
        String baseUrl = normalizeBaseUrl(StrUtil.blankToDefault(apiConfig != null ? apiConfig.getApiUrl() : null,
                DEFAULT_BASE_URL));
        if (endsWithIgnoreCase(baseUrl, "/v1/videos")) {
            return baseUrl.substring(0, baseUrl.length() - "/v1/videos".length());
        }
        if (endsWithIgnoreCase(baseUrl, "/videos")) {
            return baseUrl.substring(0, baseUrl.length() - "/videos".length());
        }
        if (endsWithIgnoreCase(baseUrl, "/v1")) {
            return baseUrl.substring(0, baseUrl.length() - "/v1".length());
        }
        return baseUrl;
    }

    public String resolveModelCode(AiModel model) {
        return model != null && StrUtil.isNotBlank(model.getCode()) ? model.getCode() : DEFAULT_VIDEO_MODEL;
    }

    public String resolveAgnesQueryUrl(OpenAiCompatibleVideoProtocolContext context, String trackingId) {
        StringBuilder builder = new StringBuilder(resolveApiRoot(context.apiConfig()))
                .append("/agnesapi?video_id=")
                .append(URLEncoder.encode(StrUtil.blankToDefault(trackingId, ""), StandardCharsets.UTF_8));

        String modelCode = resolveModelCode(context.model());
        if (StrUtil.isNotBlank(modelCode)) {
            builder.append("&model_name=")
                    .append(URLEncoder.encode(modelCode, StandardCharsets.UTF_8));
        }
        return builder.toString();
    }

    public JsonNode readJson(String responseBody, String invalidMessage) {
        try {
            return OBJECT_MAPPER.readTree(responseBody);
        } catch (IOException e) {
            throw new BusinessException(invalidMessage + ": " + previewResponse(responseBody));
        }
    }

    public String extractErrorMessage(String responseBody) {
        if (StrUtil.isBlank(responseBody)) {
            return "响应体为空";
        }
        try {
            JsonNode root = OBJECT_MAPPER.readTree(responseBody);
            String message = extractErrorMessage(root);
            return StrUtil.isNotBlank(message) ? message : previewResponse(responseBody);
        } catch (Exception ignored) {
            return previewResponse(responseBody);
        }
    }

    public String extractErrorMessage(JsonNode root) {
        if (root == null) {
            return null;
        }
        JsonNode error = root.path("error");
        if (!error.isMissingNode() && !error.isNull()) {
            String message = firstText(error, "message", "detail", "code");
            if (StrUtil.isNotBlank(message)) {
                return message;
            }
            if (error.isTextual()) {
                return error.asText();
            }
        }
        return firstText(root, "message", "detail");
    }

    public String firstText(JsonNode node, String... fields) {
        if (node == null) {
            return null;
        }
        for (String field : fields) {
            JsonNode value = node.path(field);
            if (!value.isMissingNode() && !value.isNull()) {
                String text = value.asText();
                if (StrUtil.isNotBlank(text)) {
                    return text.trim();
                }
            }
        }
        return null;
    }

    public String normalizeStatus(String status) {
        return status == null ? "" : status.trim().toLowerCase(Locale.ROOT);
    }

    public Integer parsePositiveSeconds(String value) {
        if (StrUtil.isBlank(value)) {
            return null;
        }
        try {
            double parsed = Double.parseDouble(value.trim());
            return parsed > 0 ? (int) Math.round(parsed) : null;
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    public String getString(JSONObject config, String... keys) {
        if (config == null) {
            return null;
        }
        for (String key : keys) {
            Object value = config.get(key);
            if (value == null) {
                continue;
            }
            String text = value.toString().trim();
            if (StrUtil.isNotBlank(text)) {
                return text;
            }
        }
        return null;
    }

    public Integer getInteger(JSONObject config, String... keys) {
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

    public Long getLong(JSONObject config, String... keys) {
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

    public Integer getPositiveInteger(JSONObject config, String... keys) {
        Integer value = getInteger(config, keys);
        return value != null && value > 0 ? value : null;
    }

    public List<String> resolveAgnesImageInputs(VideoTask task) {
        Set<String> ordered = new LinkedHashSet<>();
        if (StrUtil.isNotBlank(task.getFirstFrameImageUrl())) {
            ordered.add(task.getFirstFrameImageUrl().trim());
        }
        ordered.addAll(parseJsonUrls(task.getReferenceImageUrls()));
        if (StrUtil.isNotBlank(task.getLastFrameImageUrl())) {
            ordered.add(task.getLastFrameImageUrl().trim());
        }
        return new ArrayList<>(ordered);
    }

    public String resolveSingleReferenceImageUrl(VideoTask task) {
        if (StrUtil.isNotBlank(task.getFirstFrameImageUrl())) {
            return task.getFirstFrameImageUrl();
        }
        List<String> referenceImages = parseJsonUrls(task.getReferenceImageUrls());
        return referenceImages.isEmpty() ? null : referenceImages.get(0);
    }

    public List<String> parseJsonUrls(String jsonUrls) {
        List<String> urls = new ArrayList<>();
        if (StrUtil.isBlank(jsonUrls)) {
            return urls;
        }

        String trimmed = jsonUrls.trim();
        if (!trimmed.startsWith("[")) {
            if (StrUtil.isNotBlank(trimmed)) {
                urls.add(trimmed);
            }
            return urls;
        }

        try {
            JSONArray array = JSONUtil.parseArray(trimmed);
            for (Object item : array) {
                String value = item == null ? null : item.toString().trim();
                if (StrUtil.isNotBlank(value)) {
                    urls.add(value);
                }
            }
            return urls;
        } catch (Exception e) {
            throw new BusinessException("解析参考图列表失败: " + e.getMessage());
        }
    }

    public Integer resolveDurationSeconds(VideoTask task, JSONObject modelConfig) {
        if (task.getDuration() != null && task.getDuration() > 0) {
            return task.getDuration();
        }
        Integer defaultDuration = getPositiveInteger(modelConfig, "defaultDuration", "seconds", "duration");
        return defaultDuration != null ? defaultDuration : null;
    }

    public Integer resolveAgnesFrameRate(JSONObject modelConfig) {
        Integer frameRate = getPositiveInteger(modelConfig, "frameRate", "frame_rate", "fps");
        return frameRate != null ? Math.min(frameRate, 60) : 24;
    }

    public Integer resolveAgnesNumFrames(VideoTask task, JSONObject modelConfig, Integer frameRate) {
        Integer configured = getPositiveInteger(modelConfig, "numFrames", "num_frames");
        if (configured != null) {
            return normalizeAgnesNumFrames(configured);
        }

        Integer durationSeconds = resolveDurationSeconds(task, modelConfig);
        int fps = frameRate != null && frameRate > 0 ? frameRate : 24;
        int rawFrames = durationSeconds != null && durationSeconds > 0 ? durationSeconds * fps : 121;
        return normalizeAgnesNumFrames(rawFrames);
    }

    public int[] resolveAgnesDimensions(VideoTask task, JSONObject modelConfig) {
        int[] parsed = parseDimensions(task.getResolution());
        if (parsed != null) {
            return parsed;
        }

        Integer width = getPositiveInteger(modelConfig, "width", "defaultWidth", "videoWidth", "video_width");
        Integer height = getPositiveInteger(modelConfig, "height", "defaultHeight", "videoHeight", "video_height");
        if (width != null && height != null) {
            return new int[]{width, height};
        }

        parsed = parseDimensions(getString(modelConfig, "resolution", "defaultResolution"));
        if (parsed != null) {
            return parsed;
        }

        return defaultDimensionsByRatio(task.getRatio());
    }

    public BinaryResource loadBinaryResource(String sourceUrl, ApiConfig apiConfig) throws IOException {
        if (StrUtil.isBlank(sourceUrl)) {
            throw new BusinessException("OpenAI 视频参考图地址为空");
        }
        String trimmed = sourceUrl.trim();
        if (trimmed.startsWith("data:")) {
            return parseDataUrl(trimmed);
        }
        if (trimmed.startsWith("/media/")) {
            return loadLocalMedia(trimmed);
        }
        if (presetArtStyleResourceResolver.isPresetArtStylePath(trimmed)) {
            PresetArtStyleResourceResolver.PresetArtStyleResource resource = presetArtStyleResourceResolver.load(trimmed);
            return new BinaryResource(resource.bytes(), resource.mimeType(), extensionFromImageMimeType(resource.mimeType()));
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
            OkHttpClient client = AiProxySupport.okHttpClient(resourceHttpClient, apiConfig);
            try (Response response = client.newCall(request).execute()) {
                if (!response.isSuccessful() || response.body() == null) {
                    throw new BusinessException("下载视频参考图失败: HTTP " + response.code() + " url=" + trimmed);
                }
                String mimeType = normalizeImageMimeType(response.header("Content-Type"), trimmed);
                return new BinaryResource(response.body().bytes(), mimeType, extensionFromImageMimeType(mimeType));
            }
        }

        Path localPath = Paths.get(trimmed);
        if (Files.exists(localPath) && Files.isRegularFile(localPath)) {
            return loadFile(localPath);
        }
        throw new BusinessException("视频参考图地址不可访问: " + trimmed);
    }

    public MediaType mediaTypeOrDefault(String mimeType) {
        try {
            return MediaType.get(StrUtil.blankToDefault(mimeType, "image/png"));
        } catch (Exception ignored) {
            return MediaType.get("application/octet-stream");
        }
    }

    public String extensionFromImageMimeType(String mimeType) {
        String normalized = normalizeImageMimeType(mimeType, mimeType).toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "image/jpeg" -> "jpg";
            case "image/webp" -> "webp";
            case "image/gif" -> "gif";
            default -> "png";
        };
    }

    public String extensionFromVideoMimeType(String contentType) {
        if (StrUtil.isBlank(contentType)) {
            return "mp4";
        }
        String normalized = contentType.split(";", 2)[0].trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "video/webm" -> "webm";
            case "video/quicktime" -> "mov";
            case "video/x-matroska" -> "mkv";
            default -> "mp4";
        };
    }

    private JSONObject asJsonObject(Object value) {
        if (value instanceof JSONObject jsonObject) {
            return new JSONObject(jsonObject);
        }
        if (value instanceof Map<?, ?> map) {
            JSONObject jsonObject = new JSONObject();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (entry.getKey() != null) {
                    jsonObject.set(entry.getKey().toString(), entry.getValue());
                }
            }
            return jsonObject;
        }
        return null;
    }

    private String resolveSeconds(VideoTask task, JSONObject modelConfig) {
        Integer duration = resolveDurationSeconds(task, modelConfig);
        return duration != null ? String.valueOf(duration) : null;
    }

    private String resolveSize(VideoTask task, JSONObject modelConfig) {
        String resolution = normalizeSize(task.getResolution());
        if (StrUtil.isNotBlank(resolution)) {
            return resolution;
        }
        return normalizeSize(getString(modelConfig, "size", "defaultResolution", "resolution"));
    }

    private String resolveAgnesMode(VideoTask task, JSONObject modelConfig, int imageCount) {
        String explicitMode = getString(modelConfig, "agnesMode", "mode");
        if (StrUtil.isNotBlank(explicitMode)) {
            return explicitMode;
        }

        String generateMode = StrUtil.blankToDefault(task.getGenerateMode(), "").toLowerCase(Locale.ROOT);
        if (generateMode.contains("keyframe")) {
            return "keyframes";
        }
        if (StrUtil.isNotBlank(task.getLastFrameImageUrl()) && imageCount >= 2) {
            return "keyframes";
        }
        return null;
    }

    private int normalizeAgnesNumFrames(int candidate) {
        int normalized = Math.max(candidate, 1);
        if (normalized > 441) {
            normalized = 441;
        }
        int remainder = Math.floorMod(normalized - 1, 8);
        if (remainder != 0) {
            normalized += 8 - remainder;
        }
        return Math.min(normalized, 441);
    }

    private int[] defaultDimensionsByRatio(String ratio) {
        String normalized = StrUtil.blankToDefault(ratio, "16:9")
                .trim()
                .replace('：', ':')
                .replace(" ", "");
        return switch (normalized) {
            case "9:16", "2:3" -> new int[]{768, 1152};
            case "1:1" -> new int[]{1024, 1024};
            case "4:3" -> new int[]{1024, 768};
            case "3:4" -> new int[]{768, 1024};
            case "21:9" -> new int[]{1536, 640};
            default -> new int[]{1152, 768};
        };
    }

    private int[] parseDimensions(String resolution) {
        if (StrUtil.isBlank(resolution)) {
            return null;
        }

        String normalized = resolution.trim()
                .toLowerCase(Locale.ROOT)
                .replace('*', 'x')
                .replace('×', 'x')
                .replace(" ", "");
        if (!normalized.matches("\\d+x\\d+")) {
            return null;
        }

        String[] parts = normalized.split("x", 2);
        try {
            int width = Integer.parseInt(parts[0]);
            int height = Integer.parseInt(parts[1]);
            return width > 0 && height > 0 ? new int[]{width, height} : null;
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    private String normalizeSize(String size) {
        if (StrUtil.isBlank(size)) {
            return null;
        }
        String normalized = size.trim().toLowerCase(Locale.ROOT).replace('*', 'x').replace('×', 'x');
        return normalized.matches("\\d+x\\d+") ? normalized : null;
    }

    private boolean shouldAutoAppendV1Path(ApiConfig apiConfig) {
        if (apiConfig == null) {
            return true;
        }
        return !Boolean.FALSE.equals(apiConfig.getAutoAppendV1Path());
    }

    private String normalizeBaseUrl(String baseUrl) {
        return StrUtil.blankToDefault(baseUrl, DEFAULT_BASE_URL).trim().replaceAll("/+$", "");
    }

    private boolean endsWithIgnoreCase(String text, String suffix) {
        return text != null && suffix != null && text.toLowerCase(Locale.ROOT)
                .endsWith(suffix.toLowerCase(Locale.ROOT));
    }

    private void appendOptionalString(JSONObject target, String key, String value) {
        if (target != null && StrUtil.isNotBlank(value)) {
            target.set(key, value);
        }
    }

    private void appendOptionalInteger(JSONObject target, String key, Integer value) {
        if (target != null && value != null) {
            target.set(key, value);
        }
    }

    private BinaryResource parseDataUrl(String sourceUrl) {
        int commaIndex = sourceUrl.indexOf(',');
        if (commaIndex <= 0) {
            throw new BusinessException("OpenAI 视频参考图 data URL 格式非法");
        }
        String metadata = sourceUrl.substring(0, commaIndex);
        String payload = sourceUrl.substring(commaIndex + 1);
        String mimeType = normalizeImageMimeType(metadata.substring("data:".length()), sourceUrl);
        try {
            byte[] bytes = Base64.getDecoder().decode(payload.getBytes(StandardCharsets.UTF_8));
            return new BinaryResource(bytes, mimeType, extensionFromImageMimeType(mimeType));
        } catch (IllegalArgumentException e) {
            throw new BusinessException("OpenAI 视频参考图 data URL base64 非法: " + e.getMessage());
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
        throw new BusinessException("本地视频参考图不存在: " + sourceUrl);
    }

    private BinaryResource loadFile(Path path) throws IOException {
        String mimeType = normalizeImageMimeType(null, path.getFileName().toString());
        return new BinaryResource(Files.readAllBytes(path), mimeType, extensionFromImageMimeType(mimeType));
    }

    private String normalizeImageMimeType(String contentType, String sourceUrl) {
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

    private String previewResponse(String responseBody) {
        if (StrUtil.isBlank(responseBody)) {
            return "<empty>";
        }
        String normalized = responseBody.replaceAll("\\s+", " ").trim();
        return normalized.length() <= 240 ? normalized : normalized.substring(0, 240) + "...";
    }

    public record BinaryResource(byte[] bytes, String mimeType, String extension) {
    }
}
