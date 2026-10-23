package com.stonewu.fusion.service.generation.video.consumer;

import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.media.ProviderMediaFetcher;
import com.stonewu.fusion.service.media.MediaRemoteAddressPolicy;
import com.stonewu.fusion.service.media.model.MediaException;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicInteger;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 供应商下载已移入统一媒体接收器，此处保留真实 HTTP 的凭据边界验证。 */
class ProtectedVideoPersistenceTests {
    private final ProviderMediaFetcher fetcher = new ProviderMediaFetcher(new MediaRemoteAddressPolicy());

    @Test
    void downloadsProtectedVideoContentWithProviderBearerToken() throws Exception {
        byte[] payload = "protected-video-bytes".getBytes(StandardCharsets.UTF_8);
        var authorization = new AtomicReference<String>();
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/v1/videos/task/content", exchange -> {
            authorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
            exchange.sendResponseHeaders(200, payload.length);
            try (var output = exchange.getResponseBody()) { output.write(payload); }
        });
        server.start();
        var file = Files.createTempFile("provider-auth-", ".media");
        try {
            String base = "http://localhost:" + server.getAddress().getPort();
            fetcher.copy(base + "/v1/videos/task/content", ApiConfig.builder().apiUrl(base).apiKey("test-key").build(), file);
            assertThat(authorization.get()).isEqualTo("Bearer test-key");
            assertThat(Files.readAllBytes(file)).isEqualTo(payload);
        } finally { Files.deleteIfExists(file); server.stop(0); }
    }

    @Test
    void redirectCannotForwardCredentialToAnotherPrivateOrigin() throws Exception {
        var reached = new AtomicInteger();
        HttpServer target = HttpServer.create(new InetSocketAddress(0), 0);
        target.createContext("/secret", exchange -> { reached.incrementAndGet(); exchange.sendResponseHeaders(200, -1); exchange.close(); });
        target.start();
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/redirect", exchange -> {
            exchange.getResponseHeaders().add("Location", "http://localhost:" + target.getAddress().getPort() + "/secret");
            exchange.sendResponseHeaders(302, -1); exchange.close();
        });
        server.start();
        var file = Files.createTempFile("provider-redirect-", ".media");
        try {
            String base = "http://localhost:" + server.getAddress().getPort();
            assertThatThrownBy(() -> fetcher.copy(base + "/redirect", ApiConfig.builder().apiUrl(base).apiKey("secret").build(), file))
                    .isInstanceOf(MediaException.class);
            assertThat(reached).hasValue(0);
        } finally { Files.deleteIfExists(file); server.stop(0); target.stop(0); }
    }

    @Test
    void invalidBinaryDataAndLocalProviderPathsAreRejected() throws Exception {
        var file = Files.createTempFile("provider-invalid-", ".media");
        try {
            for (String uri : new String[]{"file:///etc/passwd", "/media/another-user.png", "data:image/png;base64,%%%", "data:text/html;base64,SGk="}) {
                assertThatThrownBy(() -> fetcher.copy(uri, null, file)).isInstanceOf(MediaException.class);
            }
        } finally { Files.deleteIfExists(file); }
    }
}
