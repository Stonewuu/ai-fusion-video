package com.stonewu.fusion.service.media;

import com.stonewu.fusion.entity.ai.ApiConfig;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ProviderMediaFetcherTests {
    @TempDir
    Path directory;

    /** 平台同域的签名下载地址也不能收到原接口密钥。 */
    @Test
    void redirectDropsAuthorizationEvenWhenTheSignedFileUsesTheSameOrigin() throws Exception {
        var headers = new CopyOnWriteArrayList<String>();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            headers.add(String.valueOf(exchange.getRequestHeaders().getFirst("Authorization")));
            if (exchange.getRequestURI().getPath().equals("/v1/tasks/123/content")) {
                exchange.getResponseHeaders().set("Location", "/signed-file?signature=test");
                exchange.sendResponseHeaders(302, -1);
            } else {
                exchange.sendResponseHeaders(200, 3);
                exchange.getResponseBody().write(new byte[]{1, 2, 3});
            }
            exchange.close();
        });
        server.start();
        try {
            var policy = mock(MediaRemoteAddressPolicy.class);
            when(policy.resolve(any(URI.class))).thenAnswer(invocation -> {
                URI uri = invocation.getArgument(0);
                return new MediaRemoteAddressPolicy.Target(uri, uri.getHost(), List.of(InetAddress.getLoopbackAddress()));
            });
            String root = "http://127.0.0.1:" + server.getAddress().getPort();
            var api = ApiConfig.builder().apiUrl(root).apiKey("private-test-key").build();
            Path file = directory.resolve("received.mp4");
            new ProviderMediaFetcher(policy).copy(root + "/v1/tasks/123/content", api, file);
            assertThat(headers).containsExactly("Bearer private-test-key", "null");
            assertThat(Files.readAllBytes(file)).containsExactly((byte) 1, (byte) 2, (byte) 3);
        } finally {
            server.stop(0);
        }
    }
}
