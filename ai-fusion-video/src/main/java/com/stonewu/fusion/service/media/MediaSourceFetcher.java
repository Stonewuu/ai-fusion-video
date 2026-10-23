package com.stonewu.fusion.service.media;

import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.storage.LegacyMediaRoot;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.net.Proxy;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

/**
 * 只读取已授权业务字段；逐跳检查并固定 DNS 结果，不转发认证头或走隐式代理。
 */
@Component
public class MediaSourceFetcher {
    private static final long MAX_BYTES = MediaIntakeService.MAX_RESULT_BYTES;
    private final LegacyMediaRoot localRoot;
    private final MediaRemoteAddressPolicy addresses;
    private final OkHttpClient client;

    @Autowired
    public MediaSourceFetcher(LegacyMediaRoot localRoot, MediaRemoteAddressPolicy addresses) {
        this(localRoot, addresses, new OkHttpClient.Builder().followRedirects(false).followSslRedirects(false)
                .proxy(Proxy.NO_PROXY).connectTimeout(Duration.ofSeconds(15)).readTimeout(Duration.ofSeconds(30))
                .callTimeout(Duration.ofMinutes(2)).build());
    }

    MediaSourceFetcher(LegacyMediaRoot localRoot, MediaRemoteAddressPolicy addresses, OkHttpClient client) {
        this.localRoot = localRoot;
        this.addresses = addresses;
        this.client = client;
    }

    public void copy(String source, Path output) throws IOException {
        copy(source, output, MediaSourceFetcher::requireNotInterrupted);
    }

    public void copy(String source, Path output, Runnable check) throws IOException {
        check.run();
        URI uri;
        try {
            uri = new URI(source);
        } catch (URISyntaxException invalid) {
            throw MediaException.invalid("媒体来源地址无效");
        }
        if (uri.getScheme() == null) {
            local(uri, output, check);
            return;
        }
        long deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(2);
        for (int hop = 0; hop <= 5; hop++) {
            check.run();
            var target = addresses.resolve(uri);
            long remaining = deadline - System.nanoTime();
            if (remaining <= 0) {
                throw new IOException("媒体下载超时");
            }
            var pinned = client.newBuilder().proxy(Proxy.NO_PROXY).followRedirects(false).followSslRedirects(false)
                    .callTimeout(Duration.ofNanos(remaining)).dns(host -> {
                        if (!host.equalsIgnoreCase(target.host())) {
                            throw new UnknownHostException("下载主机与固定地址不一致");
                        }
                        return target.addresses();
                    }).build();
            try (var response = pinned.newCall(new Request.Builder().url(uri.toString()).get()
                    .header("Accept-Encoding", "identity").build()).execute()) {
                int code = response.code();
                if (code == 301 || code == 302 || code == 303 || code == 307 || code == 308) {
                    String location = response.header("Location");
                    if (location == null || hop == 5) {
                        throw new IOException("媒体重定向次数或地址无效");
                    }
                    try {
                        uri = uri.resolve(new URI(location));
                    } catch (URISyntaxException invalid) {
                        throw MediaException.invalid("媒体重定向地址无效");
                    }
                    continue;
                }
                if (code != 200 || response.body() == null) {
                    throw new IOException("媒体来源未返回完整文件");
                }
                if (response.body().contentLength() > MAX_BYTES) {
                    throw tooLarge();
                }
                try (var input = response.body().byteStream()) {
                    limitedCopy(input, output, check);
                }
                return;
            }
        }
        throw new IOException("媒体来源未返回可用内容");
    }

    private void local(URI uri, Path output, Runnable check) throws IOException {
        if (uri.getRawAuthority() != null || uri.getRawQuery() != null || uri.getRawFragment() != null
                || uri.getPath() == null || !uri.getPath().startsWith("/media/")) {
            throw MediaException.invalid("不支持此本地媒体来源路径");
        }
        String relative = uri.getPath().substring("/media/".length());
        if (relative.contains("\\") || relative.contains(":") || relative.contains("\0")) {
            throw MediaException.invalid("本地媒体路径无效");
        }
        Path base = localRoot.path().toRealPath();
        Path resolved = base.resolve(relative).normalize();
        if (!resolved.startsWith(base) || relative.toLowerCase(Locale.ROOT).contains(".afv-managed")) {
            throw MediaException.invalid("媒体路径不在历史资源目录中");
        }
        Path real = resolved.toRealPath();
        if (!real.startsWith(base) || !Files.isRegularFile(real)) {
            throw MediaException.invalid("本地媒体来源无效");
        }
        if (Files.size(real) > MAX_BYTES) {
            throw tooLarge();
        }
        try (var input = Files.newInputStream(real)) {
            limitedCopy(input, output, check);
        }
    }

    private void limitedCopy(InputStream input, Path output, Runnable check) throws IOException {
        try (var destination = Files.newOutputStream(output)) {
            long total = 0;
            long nextCheck = 0;
            byte[] buffer = new byte[65536];
            for (int count; (count = input.read(buffer)) != -1; ) {
                requireNotInterrupted();
                if (System.nanoTime() >= nextCheck) {
                    check.run();
                    nextCheck = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(500);
                }
                total += count;
                if (total > MAX_BYTES) {
                    throw tooLarge();
                }
                destination.write(buffer, 0, count);
            }
            check.run();
        }
    }

    private static void requireNotInterrupted() {
        if (Thread.currentThread().isInterrupted()) {
            throw new MediaException(409, "MEDIA_IMPORT_INTERRUPTED", "媒体读取已中断");
        }
    }

    private MediaException tooLarge() {
        return new MediaException(413, "MEDIA_TOO_LARGE", "来源媒体超过 2 GiB 限制");
    }
}
