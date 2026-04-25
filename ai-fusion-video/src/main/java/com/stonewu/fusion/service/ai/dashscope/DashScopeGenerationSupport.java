package com.stonewu.fusion.service.ai.dashscope;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.enums.ai.AiModelTypeEnum;
import com.stonewu.fusion.service.storage.StorageConfigService;
import lombok.RequiredArgsConstructor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

/**
 * DashScope / 阿里云百炼生成类接口公共工具。
 */
@Component
@RequiredArgsConstructor
public class DashScopeGenerationSupport {

    public static final String PLATFORM = "dashscope";
    public static final String DEFAULT_ROOT_URL = "https://dashscope.aliyuncs.com";
    public static final String DEFAULT_GENERATION_BASE_URL = DEFAULT_ROOT_URL + "/api/v1";
    public static final String DEFAULT_COMPATIBLE_BASE_URL = DEFAULT_ROOT_URL + "/compatible-mode/v1";

    private static final String DEFAULT_LOCAL_MEDIA_BASE_PATH = "./data/media";

    private final StorageConfigService storageConfigService;

    private final OkHttpClient downloadHttpClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .followRedirects(true)
            .build();

    public static String resolveCompatibleBaseUrl(String baseUrl) {
        String root = normalizeRootUrl(baseUrl);
        return ensureSuffix(root, "/compatible-mode/v1");
    }

    public static String resolveRootBaseUrl(String baseUrl) {
        return normalizeRootUrl(baseUrl);
    }

    public static String resolveGenerationBaseUrl(String baseUrl) {
        String root = normalizeRootUrl(baseUrl);
        return ensureSuffix(root, "/api/v1");
    }

    public static String resolveGenerationBaseUrl(ApiConfig apiConfig) {
        return resolveGenerationBaseUrl(apiConfig != null ? apiConfig.getApiUrl() : null);
    }

    public static void requireApiKey(ApiConfig apiConfig) {
        if (apiConfig == null || StrUtil.isBlank(apiConfig.getApiKey())) {
            throw new BusinessException("DashScope 模型缺少 API Key 配置");
        }
    }

    public static JSONObject parseModelConfig(AiModel model) {
        if (model == null || StrUtil.isBlank(model.getConfig())) {
            return new JSONObject();
        }
        try {
            return JSONUtil.parseObj(model.getConfig());
        } catch (Exception ignored) {
            return new JSONObject();
        }
    }

    public static List<String> parseJsonUrls(String jsonUrls) {
        if (StrUtil.isBlank(jsonUrls)) {
            return List.of();
        }
        try {
            JSONArray array = JSONUtil.parseArray(jsonUrls);
            return array.toList(String.class).stream()
                    .filter(StrUtil::isNotBlank)
                    .map(String::trim)
                    .toList();
        } catch (Exception ignored) {
            return List.of();
        }
    }

    public static Integer inferRemoteModelType(String modelCode) {
        if (StrUtil.isBlank(modelCode)) {
            return null;
        }
        String code = modelCode.trim().toLowerCase(Locale.ROOT);
        if (code.startsWith("qwen-image")
                || code.startsWith("wan2.") && code.contains("image")
                || code.startsWith("wanx") && code.contains("t2i")
                || code.startsWith("wan") && code.contains("t2i")
                || code.startsWith("z-image")) {
            return AiModelTypeEnum.IMAGE.getType();
        }
        if (code.startsWith("wan") && (code.contains("t2v") || code.contains("i2v")
                || code.contains("kf2v") || code.contains("r2v") || code.contains("videoedit")
                || code.contains("vace") || code.contains("animate"))
                || code.startsWith("pixverse/")
                || code.startsWith("kling/")
                || code.startsWith("vidu/")) {
            return AiModelTypeEnum.VIDEO.getType();
        }
        return AiModelTypeEnum.CHAT.getType();
    }

    public static boolean getBoolean(JSONObject config, String key, boolean defaultValue) {
        if (config == null || StrUtil.isBlank(key) || !config.containsKey(key)) {
            return defaultValue;
        }
        Object value = config.get(key);
        if (value instanceof Boolean bool) {
            return bool;
        }
        if (value instanceof Number number) {
            return number.intValue() != 0;
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
        return defaultValue;
    }

    public static Integer getInteger(JSONObject config, String key) {
        if (config == null || StrUtil.isBlank(key) || !config.containsKey(key)) {
            return null;
        }
        try {
            return config.getInt(key);
        } catch (Exception ignored) {
            return null;
        }
    }

    public static Long getLong(JSONObject config, String key) {
        if (config == null || StrUtil.isBlank(key) || !config.containsKey(key)) {
            return null;
        }
        Object value = config.get(key);
        if (value instanceof Number number) {
            return number.longValue();
        }
        if (value != null) {
            try {
                return Long.parseLong(value.toString());
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    public static String getString(JSONObject config, String key) {
        if (config == null || StrUtil.isBlank(key) || !config.containsKey(key)) {
            return null;
        }
        Object value = config.get(key);
        if (value == null) {
            return null;
        }
        String text = value.toString().trim();
        return text.isEmpty() ? null : text;
    }

    public static String normalizeResolutionLabel(String resolution) {
        if (StrUtil.isBlank(resolution)) {
            return null;
        }
        String text = resolution.trim();
        if (text.matches("(?i)^\\d{3,4}p$")) {
            return text.substring(0, text.length() - 1) + "P";
        }
        return text;
    }

    public String toMediaUrl(String sourceUrl) throws IOException {
        if (StrUtil.isBlank(sourceUrl)) {
            throw new BusinessException("DashScope 参考媒体地址为空");
        }
        String trimmed = sourceUrl.trim();
        if (trimmed.startsWith("http://") || trimmed.startsWith("https://") || trimmed.startsWith("data:")) {
            return trimmed;
        }

        BinaryResource resource;
        if (trimmed.startsWith("/media/")) {
            resource = loadLocalMedia(trimmed);
        } else if (trimmed.startsWith("file:")) {
            resource = loadFile(Paths.get(URI.create(trimmed)));
        } else {
            Path path = Paths.get(trimmed);
            if (!Files.exists(path) || !Files.isRegularFile(path)) {
                return trimmed;
            }
            resource = loadFile(path);
        }
        return "data:" + resource.mimeType() + ";base64," + Base64.getEncoder().encodeToString(resource.bytes());
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
        throw new BusinessException("本地参考媒体不存在: " + sourceUrl);
    }

    private BinaryResource loadFile(Path path) throws IOException {
        return new BinaryResource(Files.readAllBytes(path), normalizeMimeType(null, path.getFileName().toString()));
    }

    public BinaryResource download(String sourceUrl) throws IOException {
        Request request = new Request.Builder()
                .url(sourceUrl)
                .get()
                .addHeader("Accept", "image/*,video/*,audio/*,*/*;q=0.8")
                .build();
        try (Response response = downloadHttpClient.newCall(request).execute()) {
            if (!response.isSuccessful() || response.body() == null) {
                throw new BusinessException("下载参考媒体失败: HTTP " + response.code() + " url=" + sourceUrl);
            }
            return new BinaryResource(response.body().bytes(), normalizeMimeType(response.header("Content-Type"), sourceUrl));
        }
    }

    public static String normalizeMimeType(String contentType, String sourceUrl) {
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
        if (lower.endsWith(".mp4")) {
            return "video/mp4";
        }
        if (lower.endsWith(".mov")) {
            return "video/quicktime";
        }
        if (lower.endsWith(".webm")) {
            return "video/webm";
        }
        if (lower.endsWith(".mp3")) {
            return "audio/mpeg";
        }
        if (lower.endsWith(".wav")) {
            return "audio/wav";
        }
        if (lower.endsWith(".m4a")) {
            return "audio/mp4";
        }
        return "image/png";
    }

    private static String normalizeRootUrl(String baseUrl) {
        String normalized = StrUtil.isBlank(baseUrl) ? DEFAULT_ROOT_URL : baseUrl.trim().replaceAll("/+$", "");
        normalized = normalized.replaceAll("(?i)/compatible-mode/v1$", "");
        normalized = normalized.replaceAll("(?i)/api/v1$", "");
        return normalized;
    }

    private static String ensureSuffix(String root, String suffix) {
        String normalizedRoot = StrUtil.blankToDefault(root, DEFAULT_ROOT_URL).replaceAll("/+$", "");
        String normalizedSuffix = suffix.startsWith("/") ? suffix : "/" + suffix;
        return normalizedRoot + normalizedSuffix;
    }

    public record BinaryResource(byte[] bytes, String mimeType) {
    }
}
