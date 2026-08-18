package com.stonewu.fusion.service.ai.sonilo;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.SoniloBgmProperties;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * 使用 JDK 内置 HttpServer 模拟 Sonilo 接口，不引入额外测试依赖。
 */
class SoniloAudioClientTests {

    private HttpServer server;
    private SoniloBgmProperties properties;
    private SoniloAudioClient client;
    private Path tempDir;

    @BeforeEach
    void setUp() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.start();
        properties = new SoniloBgmProperties();
        properties.setEnabled(true);
        properties.setApiKey("sk-test");
        properties.setBaseUrl("http://127.0.0.1:" + server.getAddress().getPort());
        properties.setPollIntervalSeconds(0);
        properties.setTimeoutSeconds(10);
        client = new SoniloAudioClient(properties);
        tempDir = Files.createTempDirectory("sonilo_client_test_");
    }

    @AfterEach
    void tearDown() throws IOException {
        server.stop(0);
        try (var paths = Files.walk(tempDir)) {
            paths.sorted((a, b) -> b.compareTo(a)).forEach(path -> path.toFile().delete());
        }
    }

    @Test
    void submitMusicTaskSendsAsyncModeWithAuthAndPrompt() throws IOException {
        AtomicReference<String> authHeader = new AtomicReference<>();
        AtomicReference<String> userAgent = new AtomicReference<>();
        AtomicReference<String> requestBody = new AtomicReference<>();
        server.createContext("/v1/video-to-music", exchange -> {
            authHeader.set(exchange.getRequestHeaders().getFirst("Authorization"));
            userAgent.set(exchange.getRequestHeaders().getFirst("User-Agent"));
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.ISO_8859_1));
            respond(exchange, 200, "{\"task_id\":\"task-music-1\"}");
        });
        Path video = Files.write(tempDir.resolve("composed.mp4"), new byte[]{1, 2, 3});

        String taskId = client.submitMusicTask(video, "轻快的都市剧配乐");

        assertThat(taskId).isEqualTo("task-music-1");
        assertThat(authHeader.get()).isEqualTo("Bearer sk-test");
        assertThat(userAgent.get()).startsWith("ai-fusion-video");
        assertThat(requestBody.get())
                .contains("name=\"mode\"")
                .contains("async")
                .contains("name=\"prompt\"")
                .contains("name=\"video\"")
                .contains("filename=\"composed.mp4\"");
    }

    @Test
    void submitMusicTaskOmitsPromptWhenBlank() throws IOException {
        AtomicReference<String> requestBody = new AtomicReference<>();
        server.createContext("/v1/video-to-music", exchange -> {
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.ISO_8859_1));
            respond(exchange, 200, "{\"task_id\":\"task-music-2\"}");
        });
        Path video = Files.write(tempDir.resolve("composed.mp4"), new byte[]{1});

        client.submitMusicTask(video, "  ");

        assertThat(requestBody.get()).doesNotContain("name=\"prompt\"");
    }

    @Test
    void submitTaskTranslatesInsufficientCredits() throws IOException {
        server.createContext("/v1/video-to-sfx", exchange ->
                respond(exchange, 402, "{\"detail\":\"credit balance too low\"}"));
        Path video = Files.write(tempDir.resolve("composed.mp4"), new byte[]{1});

        assertThatThrownBy(() -> client.submitSfxTask(video, null))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("余额不足");
    }

    @Test
    void awaitTaskPollsUntilSucceededAndExtractsTrack() {
        AtomicInteger polls = new AtomicInteger();
        server.createContext("/v1/tasks/task-1", exchange -> {
            if (polls.incrementAndGet() < 3) {
                respond(exchange, 200, "{\"status\":\"processing\"}");
            } else {
                respond(exchange, 200, "{\"status\":\"succeeded\",\"audio\":"
                        + "{\"url\":\"http://cdn.example/track.m4a?sig=abc\",\"license_id\":\"lic-42\"}}");
            }
        });

        var task = client.awaitTask("task-1");
        SoniloAudioTrack track = client.extractPrimaryTrack(task, "task-1");

        assertThat(polls.get()).isEqualTo(3);
        assertThat(track.url()).isEqualTo("http://cdn.example/track.m4a?sig=abc");
        assertThat(track.licenseId()).isEqualTo("lic-42");
    }

    @Test
    void awaitTaskFailedThrowsReadableMessageWithRefundNote() {
        server.createContext("/v1/tasks/task-2", exchange ->
                respond(exchange, 200, "{\"status\":\"failed\",\"refunded\":true,"
                        + "\"error\":{\"message\":\"video decode error\"}}"));

        assertThatThrownBy(() -> client.awaitTask("task-2"))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("video decode error")
                .hasMessageContaining("费用已自动退还");
    }

    @Test
    void extractPrimaryTrackRejectsResultWithoutAudio() {
        server.createContext("/v1/tasks/task-3", exchange ->
                respond(exchange, 200, "{\"status\":\"succeeded\"}"));

        var task = client.awaitTask("task-3");

        assertThatThrownBy(() -> client.extractPrimaryTrack(task, "task-3"))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("未返回音轨结果");
    }

    @Test
    void downloadAudioDoesNotSendApiKeyToPresignedUrl() throws IOException {
        List<String> authHeaders = new ArrayList<>();
        byte[] audioBytes = {9, 8, 7, 6};
        server.createContext("/download/track.m4a", exchange -> {
            authHeaders.add(exchange.getRequestHeaders().getFirst("Authorization"));
            respond(exchange, 200, audioBytes);
        });
        Path dest = tempDir.resolve("track.m4a");

        client.downloadAudio(properties.getBaseUrl() + "/download/track.m4a?sig=presigned", dest);

        assertThat(authHeaders).containsExactly((String) null);
        assertThat(Files.readAllBytes(dest)).isEqualTo(audioBytes);
    }

    private void respond(HttpExchange exchange, int status, String body) throws IOException {
        respond(exchange, status, body.getBytes(StandardCharsets.UTF_8));
    }

    private void respond(HttpExchange exchange, int status, byte[] body) throws IOException {
        exchange.sendResponseHeaders(status, body.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(body);
        }
        exchange.close();
    }
}
