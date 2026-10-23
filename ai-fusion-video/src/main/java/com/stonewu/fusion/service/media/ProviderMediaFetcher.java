package com.stonewu.fusion.service.media;

import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.ProviderMediaAuthorization;
import com.stonewu.fusion.service.media.model.MediaException;
import okhttp3.HttpUrl;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.net.Proxy;
import java.net.URI;
import java.net.UnknownHostException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Base64;
import java.util.concurrent.TimeUnit;

/**
 * 仅用于可信生成任务的供应商结果；用户业务字段导入仍由公共地址策略处理。
 */
@Component
public class ProviderMediaFetcher {
    private final MediaRemoteAddressPolicy addresses;
    private final OkHttpClient client = new OkHttpClient.Builder().proxy(Proxy.NO_PROXY)
            .followRedirects(false).followSslRedirects(false)
            .connectTimeout(Duration.ofSeconds(20)).readTimeout(Duration.ofSeconds(60)).build();

    public ProviderMediaFetcher(MediaRemoteAddressPolicy addresses) {
        this.addresses = addresses;
    }

    public void copy(String source, ApiConfig config, Path output) throws IOException {
        copy(source, config, output, true);
    }

    public void copyPublic(String source, ApiConfig config, Path output) throws IOException {
        copy(source, config, output, false);
    }

    private void copy(String source, ApiConfig config, Path output, boolean allowAuthorization) throws IOException {
        if (source == null || source.isBlank()) {
            throw MediaException.invalid("供应商媒体地址为空");
        }
        if (source.startsWith("data:")) {
            data(source, output);
            return;
        }
        URI uri;
        try {
            uri = URI.create(source);
        } catch (IllegalArgumentException failure) {
            throw MediaException.invalid("供应商媒体地址无效");
        }
        long deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(10);
        for (int hop = 0; hop <= 5; hop++) {
            if (uri.getHost() == null || uri.getRawUserInfo() != null || uri.getRawFragment() != null
                    || (!"https".equalsIgnoreCase(uri.getScheme()) && !"http".equalsIgnoreCase(uri.getScheme()))) {
                throw MediaException.invalid("供应商结果必须是 HTTP(S) 媒体地址");
            }
            boolean authorized = allowAuthorization && ProviderMediaAuthorization.shouldAuthorize(uri.toString(),
                    config, defaultUrl(config));
            // 配置中的实际提供商允许使用其指定网络位置；其他目标必须是公共地址。
            var target = authorized ? null : addresses.resolve(uri);
            long remaining = deadline - System.nanoTime();
            if (remaining <= 0) {
                throw new IOException("供应商媒体下载超时");
            }
            var builder = AiProxySupport.okHttpClient(client, config).newBuilder()
                    .followRedirects(false).followSslRedirects(false).callTimeout(Duration.ofNanos(remaining));
            if (target != null) {
                builder.dns(host -> {
                    if (!host.equalsIgnoreCase(target.host())) {
                        throw new UnknownHostException("下载目标与已验证地址不符");
                    }
                    return target.addresses();
                });
            }
            var request = new Request.Builder().url(HttpUrl.get(uri.toString())).header("Accept-Encoding", "identity");
            if (authorized) {
                request.header("Authorization", "Bearer " + config.getApiKey());
            }
            try (var response = builder.build().newCall(request.get().build()).execute()) {
                int status = response.code();
                if (status == 301 || status == 302 || status == 303 || status == 307 || status == 308) {
                    if (hop == 5 || response.header("Location") == null) {
                        throw new IOException("供应商重定向无效");
                    }
                    uri = uri.resolve(response.header("Location"));
                    continue;
                }
                if (status != 200 || response.body() == null) {
                    throw new IOException("供应商媒体下载失败，HTTP " + status);
                }
                if (response.body().contentLength() > MediaIntakeService.MAX_RESULT_BYTES) {
                    throw tooLarge();
                }
                try (var input = response.body().byteStream(); var destination = Files.newOutputStream(output)) {
                    byte[] buffer = new byte[65536];
                    long total = 0;
                    for (int read; (read = input.read(buffer)) != -1; ) {
                        total += read;
                        if (total > MediaIntakeService.MAX_RESULT_BYTES) {
                            throw tooLarge();
                        }
                        destination.write(buffer, 0, read);
                    }
                }
                return;
            }
        }
        throw new IOException("供应商媒体没有可读取内容");
    }

    private void data(String source, Path output) throws IOException {
        int comma = source.indexOf(',');
        if (comma < 0 || !source.substring(0, comma).matches("data:(image|audio|video)/[A-Za-z0-9.+-]+;base64")) {
            throw MediaException.invalid("生成媒体 Data URI 格式无效");
        }
        String encoded = source.substring(comma + 1);
        if (encoded.length() > (MediaIntakeService.MAX_UPLOAD_BYTES + 2) / 3 * 4) {
            throw tooLarge();
        }
        try {
            Files.write(output, Base64.getDecoder().decode(encoded));
        } catch (IllegalArgumentException invalid) {
            throw MediaException.invalid("生成媒体 Base64 无效");
        }
    }

    private String defaultUrl(ApiConfig config) {
        if (config == null || config.getPlatform() == null) {
            return null;
        }
        return switch (config.getPlatform()) {
            case "openai", "openai_compatible", "newapi", "agnes" -> "https://api.openai.com";
            default -> null;
        };
    }

    private MediaException tooLarge() {
        return new MediaException(413, "MEDIA_TOO_LARGE", "生成结果超过媒体接收限制");
    }
}
