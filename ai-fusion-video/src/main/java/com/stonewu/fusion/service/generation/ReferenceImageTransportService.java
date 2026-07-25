package com.stonewu.fusion.service.generation;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import com.stonewu.fusion.service.system.SystemConfigService;
import lombok.RequiredArgsConstructor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/** Resolves reference images to the exact transport formats declared by a model. */
@Service
@RequiredArgsConstructor
public class ReferenceImageTransportService {

    public static final String FORMAT_URL = "url";
    public static final String FORMAT_DATA_URI = "data_uri";
    private static final String DEFAULT_LOCAL_MEDIA_BASE_PATH = "./data/media";

    private final StorageConfigService storageConfigService;
    private final SystemConfigService systemConfigService;
    private final PresetArtStyleResourceResolver presetArtStyleResourceResolver;

    private final OkHttpClient httpClient = new OkHttpClient.Builder()
            .connectTimeout(1, TimeUnit.MINUTES)
            .readTimeout(25, TimeUnit.MINUTES)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build();

    public ReferenceImageTransportCapability resolveCapability(JSONObject modelConfig) {
        Set<String> formats = new LinkedHashSet<>();
        if (modelConfig != null) {
            Object configuredFormats = modelConfig.get("referenceImageInputFormats");
            if (configuredFormats instanceof JSONArray array) {
                array.toList(String.class).forEach(value -> addNormalizedFormat(formats, value));
            } else if (configuredFormats instanceof Iterable<?> iterable) {
                for (Object value : iterable) {
                    if (value != null) addNormalizedFormat(formats, value.toString());
                }
            } else if (configuredFormats != null) {
                addNormalizedFormat(formats, configuredFormats.toString());
            }
            if (Boolean.TRUE.equals(readBoolean(modelConfig, "supportDataUriInput"))) {
                formats.add(FORMAT_DATA_URI);
            }
        }
        List<String> normalizedFormats = List.copyOf(formats);
        return new ReferenceImageTransportCapability(
                normalizedFormats,
                normalizedFormats.contains(FORMAT_URL),
                normalizedFormats.contains(FORMAT_DATA_URI));
    }

    public void validateInputs(AiModel model, JSONObject modelConfig, List<String> sources) {
        if (sources == null || sources.isEmpty()) return;
        ReferenceImageTransportCapability capability = resolveCapability(modelConfig);
        for (String source : sources) {
            selectTransport(model, capability, source);
        }
    }

    public List<String> resolveInputs(AiModel model,
                                      JSONObject modelConfig,
                                      List<String> sources,
                                      ApiConfig apiConfig) {
        if (sources == null || sources.isEmpty()) return List.of();
        ReferenceImageTransportCapability capability = resolveCapability(modelConfig);
        List<String> resolved = new ArrayList<>();
        for (String source : sources) {
            TransportSelection selection = selectTransport(model, capability, source);
            if (selection.transport() == ReferenceImageTransport.URL) {
                resolved.add(selection.value());
            } else {
                resolved.add(toDataUri(selection.value(), apiConfig));
            }
        }
        return List.copyOf(resolved);
    }

    public List<String> parseJsonInputs(String json, String fieldName) {
        if (StrUtil.isBlank(json)) return List.of();
        try {
            return JSONUtil.parseArray(json).toList(String.class).stream()
                    .filter(StrUtil::isNotBlank)
                    .map(String::trim)
                    .toList();
        } catch (Exception e) {
            throw new BusinessException(fieldName + " 必须是合法的图片地址 JSON 数组");
        }
    }

    private TransportSelection selectTransport(AiModel model,
                                                ReferenceImageTransportCapability capability,
                                                String source) {
        String normalizedSource = StrUtil.trim(source);
        if (StrUtil.isBlank(normalizedSource)) {
            throw unsupported(model, "参考图地址为空");
        }
        if (!capability.supportsAny()) {
            throw unsupported(model, "未配置允许的参考图传递模式，请在模型能力中启用 URL 或 base64/Data URI");
        }

        if (StrUtil.startWithIgnoreCase(normalizedSource, "data:")) {
            if (!capability.supportsDataUri()) {
                throw unsupported(model, "当前参考图是 Data URI，但模型未启用 base64/Data URI 传递模式");
            }
            return new TransportSelection(ReferenceImageTransport.DATA_URI, normalizedSource);
        }

        if (isHttpUrl(normalizedSource)) {
            if (capability.supportsUrl()) {
                return new TransportSelection(ReferenceImageTransport.URL, normalizedSource);
            }
            if (capability.supportsDataUri()) {
                return new TransportSelection(ReferenceImageTransport.DATA_URI, normalizedSource);
            }
        } else {
            if (capability.supportsUrl()) {
                String publicUrl = systemConfigService.resolvePublicUrl(normalizedSource);
                if (StrUtil.isNotBlank(publicUrl)) {
                    return new TransportSelection(ReferenceImageTransport.URL, publicUrl);
                }
            }
            if (capability.supportsDataUri()) {
                return new TransportSelection(ReferenceImageTransport.DATA_URI, normalizedSource);
            }
        }

        String reason = capability.supportsUrl()
                ? "模型仅允许 URL，但项目未配置访问域名或公网对象存储，且当前素材没有公网 URL"
                : "当前参考图需要 URL 传递，但模型只允许 base64/Data URI";
        throw unsupported(model, reason);
    }

    private String toDataUri(String source, ApiConfig apiConfig) {
        if (StrUtil.startWithIgnoreCase(source, "data:")) return source;
        try {
            BinaryResource resource = loadBinaryResource(source, apiConfig);
            return "data:" + resource.mimeType() + ";base64,"
                    + Base64.getEncoder().encodeToString(resource.bytes());
        } catch (IOException e) {
            throw new BusinessException("参考图转换为 base64/Data URI 失败: " + e.getMessage());
        }
    }

    private BinaryResource loadBinaryResource(String source, ApiConfig apiConfig) throws IOException {
        if (source.startsWith("/media/")) return loadLocalMedia(source);
        if (presetArtStyleResourceResolver.isPresetArtStylePath(source)) {
            var resource = presetArtStyleResourceResolver.load(source);
            return new BinaryResource(resource.bytes(), normalizeMimeType(resource.mimeType(), source));
        }
        if (isHttpUrl(source)) {
            Request request = new Request.Builder().url(source).get()
                    .addHeader("Accept", "image/*,*/*;q=0.8").build();
            OkHttpClient client = apiConfig == null
                    ? httpClient : AiProxySupport.okHttpClient(httpClient, apiConfig);
            try (Response response = client.newCall(request).execute()) {
                if (!response.isSuccessful() || response.body() == null) {
                    throw new BusinessException("下载参考图失败: HTTP " + response.code());
                }
                return new BinaryResource(response.body().bytes(),
                        normalizeMimeType(response.header("Content-Type"), source));
            }
        }
        throw new BusinessException("参考图地址不可访问: " + source);
    }

    private BinaryResource loadLocalMedia(String source) throws IOException {
        String relativePath = source.replaceFirst("^/media/?", "");
        List<Path> candidates = new ArrayList<>();
        StorageConfig config = storageConfigService.getDefaultConfig();
        if (config != null && StrUtil.isNotBlank(config.getBasePath())) {
            Path candidate = resolveLocalMediaPath(config.getBasePath(), relativePath);
            if (candidate != null) candidates.add(candidate);
        }
        Path defaultCandidate = resolveLocalMediaPath(DEFAULT_LOCAL_MEDIA_BASE_PATH, relativePath);
        if (defaultCandidate != null) candidates.add(defaultCandidate);
        for (Path candidate : candidates) {
            if (Files.exists(candidate) && Files.isRegularFile(candidate)) return loadFile(candidate);
        }
        throw new BusinessException("本地参考图不存在: " + source);
    }

    private Path resolveLocalMediaPath(String basePath, String relativePath) {
        Path root = Paths.get(basePath).toAbsolutePath().normalize();
        Path candidate = root.resolve(relativePath).normalize();
        return candidate.startsWith(root) ? candidate : null;
    }

    private BinaryResource loadFile(Path path) throws IOException {
        return new BinaryResource(Files.readAllBytes(path), normalizeMimeType(null, path.toString()));
    }

    private String normalizeMimeType(String contentType, String source) {
        if (StrUtil.isNotBlank(contentType)) return contentType.split(";", 2)[0].trim();
        String lower = StrUtil.blankToDefault(source, "").toLowerCase(Locale.ROOT);
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
        if (lower.endsWith(".webp")) return "image/webp";
        if (lower.endsWith(".gif")) return "image/gif";
        return "image/png";
    }

    private Boolean readBoolean(JSONObject config, String key) {
        if (config == null || !config.containsKey(key)) return null;
        Object value = config.get(key);
        if (value instanceof Boolean bool) return bool;
        return value == null ? null : Boolean.parseBoolean(value.toString());
    }

    private void addNormalizedFormat(Set<String> formats, String value) {
        String normalized = StrUtil.blankToDefault(value, "").trim().toLowerCase(Locale.ROOT);
        if ("base64".equals(normalized) || "data-uri".equals(normalized)) normalized = FORMAT_DATA_URI;
        if (FORMAT_URL.equals(normalized) || FORMAT_DATA_URI.equals(normalized)) formats.add(normalized);
    }

    private BusinessException unsupported(AiModel model, String reason) {
        String modelName = model == null
                ? "当前模型" : StrUtil.blankToDefault(model.getName(), model.getCode());
        return new BusinessException("模型 " + modelName + " 无法使用参考图：" + reason + "。");
    }

    private boolean isHttpUrl(String value) {
        return StrUtil.startWithIgnoreCase(value, "http://")
                || StrUtil.startWithIgnoreCase(value, "https://");
    }

    public record ReferenceImageTransportCapability(List<String> formats,
                                                    boolean supportsUrl,
                                                    boolean supportsDataUri) {
        public boolean supportsAny() {
            return supportsUrl || supportsDataUri;
        }
    }

    private enum ReferenceImageTransport {
        URL,
        DATA_URI
    }

    private record TransportSelection(ReferenceImageTransport transport, String value) {
    }

    private record BinaryResource(byte[] bytes, String mimeType) {
    }
}
