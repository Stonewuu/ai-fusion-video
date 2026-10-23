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
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.job.RemoteGenerationJob;
import com.stonewu.fusion.service.media.MediaInputTransport;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import com.stonewu.fusion.service.system.SystemConfigService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.HttpUrl;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.Inet6Address;
import java.net.InetAddress;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Resolves reference images to the exact transport formats declared by a model.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class ReferenceImageTransportService {

    public static final String FORMAT_URL = "url";
    public static final String FORMAT_DATA_URI = "data_uri";
    private static final String DEFAULT_LOCAL_MEDIA_BASE_PATH = "./data/media";

    private final StorageConfigService storageConfigService;
    private final SystemConfigService systemConfigService;
    private final PresetArtStyleResourceResolver presetArtStyleResourceResolver;
    private final MediaInputTransport media;

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
                    if (value != null) {
                        addNormalizedFormat(formats, value.toString());
                    }
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
        if (sources == null || sources.isEmpty()) {
            return;
        }
        ReferenceImageTransportCapability capability = resolveCapability(modelConfig);
        for (String source : sources) {
            selectTransport(model, capability, source);
        }
    }

    public List<String> resolveInputs(AiModel model,
                                      JSONObject modelConfig,
                                      List<String> sources,
                                      ApiConfig apiConfig, CanvasActor actor, RemoteGenerationJob.Claim claim) {
        if (sources == null || sources.isEmpty()) {
            return List.of();
        }
        ReferenceImageTransportCapability capability = resolveCapability(modelConfig);
        List<String> resolved = new ArrayList<>();
        for (int index = 0; index < sources.size(); index++) {
            try {
                TransportSelection selection = selectTransport(model, capability, sources.get(index));
                if (MediaReference.managed(selection.value())) {
                    if (actor == null) {
                        throw new BusinessException(403, "托管媒体输入缺少经过认证的用户");
                    }
                    resolved.add(selection.transport() == ReferenceImageTransport.URL
                            ? media.generationUrl(selection.value(), actor, MediaKind.IMAGE, claim)
                            : media.dataUri(selection.value(), actor, MediaKind.IMAGE,
                            MediaIntakeService.MAX_UPLOAD_BYTES));
                } else if (selection.transport() == ReferenceImageTransport.URL) {
                    resolved.add(selection.value());
                } else {
                    resolved.add(toDataUri(selection.value(), apiConfig));
                }
            } catch (RuntimeException failure) {
                // 不记录可能含签名或凭据的原始 URL；失败原因继续由任务执行器持久化。
                log.warn("参考图处理失败，停止本次生成: model={}, referenceIndex={}, errorType={}",
                        model == null ? null : model.getCode(), index + 1, failure.getClass().getSimpleName());
                throw failure;
            }
        }
        return List.copyOf(resolved);
    }

    public List<String> parseJsonInputs(String json, String fieldName) {
        if (StrUtil.isBlank(json)) {
            return List.of();
        }
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

        if (MediaReference.managed(normalizedSource)) {
            MediaReference.id(normalizedSource);
            if (capability.supportsDataUri()) {
                return new TransportSelection(ReferenceImageTransport.DATA_URI, normalizedSource);
            }
            String base = systemConfigService.getPublicResourceBaseUrl();
            if (capability.supportsUrl() && isPotentiallyPublicHttpUrl(base)) {
                return new TransportSelection(ReferenceImageTransport.URL, normalizedSource);
            }
            throw unsupported(model, "托管媒体的 URL 传递需要配置可访问的后端资源公网地址");
        }

        if (StrUtil.startWithIgnoreCase(normalizedSource, "data:")) {
            if (!capability.supportsDataUri()) {
                throw unsupported(model, "当前参考图是 Data URI，但模型未启用 base64/Data URI 传递模式");
            }
            return new TransportSelection(ReferenceImageTransport.DATA_URI, normalizedSource);
        }

        if (isHttpUrl(normalizedSource)) {
            if (capability.supportsUrl() && isPotentiallyPublicHttpUrl(normalizedSource)) {
                return new TransportSelection(ReferenceImageTransport.URL, normalizedSource);
            }
            if (capability.supportsDataUri()) {
                return new TransportSelection(ReferenceImageTransport.DATA_URI, normalizedSource);
            }
            throw unsupported(model,
                    "当前参考图 URL 指向本机、回环或内网地址，远端模型无法访问，且模型未启用 base64/Data URI 传递模式");
        } else {
            if (capability.supportsUrl()) {
                String publicUrl = systemConfigService.resolvePublicUrl(normalizedSource);
                if (StrUtil.isNotBlank(publicUrl) && isPotentiallyPublicHttpUrl(publicUrl)) {
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
        if (StrUtil.startWithIgnoreCase(source, "data:")) {
            return source;
        }
        try {
            BinaryResource resource = loadBinaryResource(source, apiConfig);
            return "data:" + resource.mimeType() + ";base64,"
                    + Base64.getEncoder().encodeToString(resource.bytes());
        } catch (IOException e) {
            throw new BusinessException("参考图转换为 base64/Data URI 失败: " + e.getMessage());
        }
    }

    private BinaryResource loadBinaryResource(String source, ApiConfig apiConfig) throws IOException {
        source = resolveLocalResourceSource(source);
        if (source.startsWith("/media/")) {
            return loadLocalMedia(source);
        }
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

    /** 本站完整资源地址与相对路径等价，不经容器外部网关重新下载。 */
    private String resolveLocalResourceSource(String source) {
        if (!isHttpUrl(source)) {
            return source;
        }
        String configuredBase = systemConfigService.getPublicResourceBaseUrl();
        if (StrUtil.isBlank(configuredBase)) {
            return source;
        }
        HttpUrl url;
        HttpUrl base;
        try {
            url = HttpUrl.get(source);
            base = HttpUrl.get(configuredBase);
        } catch (IllegalArgumentException invalid) {
            throw new BusinessException("参考图或后端资源地址不是有效的 HTTP(S) URL");
        }
        if (!url.scheme().equals(base.scheme()) || !url.host().equals(base.host())
                || url.port() != base.port() || !url.username().isEmpty() || !url.password().isEmpty()) {
            return source;
        }
        String prefix = base.encodedPath().replaceAll("/+$", "");
        if (!url.encodedPath().startsWith(prefix + "/")) {
            return source;
        }
        String path = URI.create(url.encodedPath().substring(prefix.length())).getPath();
        return path.startsWith("/media/") || presetArtStyleResourceResolver.isPresetArtStylePath(path)
                ? path : source;
    }

    private BinaryResource loadLocalMedia(String source) throws IOException {
        String relativePath = source.replaceFirst("^/media/?", "");
        List<Path> candidates = new ArrayList<>();
        StorageConfig config = storageConfigService.getDefaultConfig();
        if (config != null && StrUtil.isNotBlank(config.getBasePath())) {
            Path candidate = resolveLocalMediaPath(config.getBasePath(), relativePath);
            if (candidate != null) {
                candidates.add(candidate);
            }
        }
        Path defaultCandidate = resolveLocalMediaPath(DEFAULT_LOCAL_MEDIA_BASE_PATH, relativePath);
        if (defaultCandidate != null) {
            candidates.add(defaultCandidate);
        }
        for (Path candidate : candidates) {
            if (Files.exists(candidate) && Files.isRegularFile(candidate)) {
                return loadFile(candidate);
            }
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
        if (StrUtil.isNotBlank(contentType)) {
            return contentType.split(";", 2)[0].trim();
        }
        String lower = StrUtil.blankToDefault(source, "").toLowerCase(Locale.ROOT);
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

    private Boolean readBoolean(JSONObject config, String key) {
        if (config == null || !config.containsKey(key)) {
            return null;
        }
        Object value = config.get(key);
        if (value instanceof Boolean bool) {
            return bool;
        }
        return value == null ? null : Boolean.parseBoolean(value.toString());
    }

    private void addNormalizedFormat(Set<String> formats, String value) {
        String normalized = StrUtil.blankToDefault(value, "").trim().toLowerCase(Locale.ROOT);
        if ("base64".equals(normalized) || "data-uri".equals(normalized)) {
            normalized = FORMAT_DATA_URI;
        }
        if (FORMAT_URL.equals(normalized) || FORMAT_DATA_URI.equals(normalized)) {
            formats.add(normalized);
        }
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

    private boolean isPotentiallyPublicHttpUrl(String value) {
        try {
            URI uri = URI.create(value);
            String host = StrUtil.blankToDefault(uri.getHost(), "").trim().toLowerCase(Locale.ROOT);
            while (host.endsWith(".")) {
                host = host.substring(0, host.length() - 1);
            }
            if (StrUtil.isBlank(host)
                    || "localhost".equals(host)
                    || host.endsWith(".localhost")
                    || host.endsWith(".local")
                    || "host.docker.internal".equals(host)
                    || "gateway.docker.internal".equals(host)) {
                return false;
            }
            if (!isIpLiteral(host)) {
                return true;
            }

            InetAddress address = InetAddress.getByName(host);
            if (address.isAnyLocalAddress()
                    || address.isLoopbackAddress()
                    || address.isLinkLocalAddress()
                    || address.isSiteLocalAddress()
                    || address.isMulticastAddress()) {
                return false;
            }
            return !(address instanceof Inet6Address) || !isUniqueLocalIpv6(address);
        } catch (Exception ignored) {
            return false;
        }
    }

    private boolean isIpLiteral(String host) {
        if (host.indexOf(':') >= 0) {
            return true;
        }
        if (host.isEmpty()) {
            return false;
        }
        for (int i = 0; i < host.length(); i++) {
            char current = host.charAt(i);
            if (current != '.' && !Character.isDigit(current)) {
                return false;
            }
        }
        return true;
    }

    private boolean isUniqueLocalIpv6(InetAddress address) {
        byte[] bytes = address.getAddress();
        return bytes.length == 16 && (bytes[0] & 0xFE) == 0xFC;
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
