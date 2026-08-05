package com.stonewu.fusion.service.ai.comfyui.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ComfyUiNativeClientTests {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private HttpServer server;
    private ApiConfig apiConfig;
    private ComfyUiNativeClient client;

    @BeforeEach
    void setUp() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.start();
        apiConfig = ApiConfig.builder()
                .apiUrl("http://127.0.0.1:" + server.getAddress().getPort())
                .apiKey("test-token")
                .build();
        client = new ComfyUiNativeClient(objectMapper);
    }

    @AfterEach
    void tearDown() {
        server.stop(0);
    }

    @Test
    void testConnectionRequiresAndRecognizesJobsApi() {
        server.createContext("/system_stats", exchange -> json(exchange, 200,
                "{\"system\":{\"comfyui_version\":\"0.30.0\"}}"));
        server.createContext("/features", exchange -> json(exchange, 200, "{\"jobs\":true}"));
        server.createContext("/api/jobs/", exchange -> json(exchange, 404,
                "{\"error\":\"Job not found\"}"));

        ComfyUiConnectionResult result = client.testConnection(apiConfig);

        assertThat(result.connected()).isTrue();
        assertThat(result.jobsApiSupported()).isTrue();
        assertThat(result.version()).isEqualTo("0.30.0");
    }

    @Test
    void testConnectionRejectsNonOfficialSystemStatsVersionField() {
        server.createContext("/system_stats", exchange -> json(exchange, 200,
                "{\"system\":{\"version\":\"0.30.0\"}}"));
        server.createContext("/features", exchange -> json(exchange, 200, "{\"jobs\":true}"));
        server.createContext("/api/jobs/", exchange -> json(exchange, 404,
                "{\"error\":\"Job not found\"}"));

        assertThatThrownBy(() -> client.testConnection(apiConfig))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("system.comfyui_version");
    }

    @Test
    void submitPromptSendsBearerTokenAndCallerPromptId() {
        String promptId = UUID.randomUUID().toString();
        AtomicReference<String> authorization = new AtomicReference<>();
        AtomicReference<JsonNode> requestJson = new AtomicReference<>();
        server.createContext("/prompt", exchange -> {
            authorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
            requestJson.set(objectMapper.readTree(exchange.getRequestBody()));
            json(exchange, 200, "{\"prompt_id\":\"" + promptId
                    + "\",\"number\":7,\"node_errors\":{}}");
        });
        ObjectNode prompt = objectMapper.createObjectNode();
        prompt.putObject("1").put("class_type", "KSampler").putObject("inputs");

        ComfyUiPromptResponse result = client.submitPrompt(apiConfig, prompt, promptId);

        assertThat(result.promptId()).isEqualTo(promptId);
        assertThat(result.number()).isEqualTo(7D);
        assertThat(authorization.get()).isEqualTo("Bearer test-token");
        assertThat(requestJson.get().path("prompt_id").asText()).isEqualTo(promptId);
        assertThat(requestJson.get().path("prompt").path("1").path("class_type").asText())
                .isEqualTo("KSampler");
    }

    @Test
    void submitPromptRejectsNonCanonicalPromptIdBeforeNetworkCall() {
        ObjectNode prompt = objectMapper.createObjectNode();
        prompt.putObject("1").put("class_type", "KSampler").putObject("inputs");

        assertThatThrownBy(() -> client.submitPrompt(apiConfig, prompt, "NOT-A-UUID"))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("canonical UUID");
    }

    @Test
    void uploadImageUsesOfficialMultipartContractAndParsesDescriptor() {
        AtomicReference<String> authorization = new AtomicReference<>();
        AtomicReference<String> contentType = new AtomicReference<>();
        AtomicReference<String> requestBody = new AtomicReference<>();
        server.createContext("/upload/image", exchange -> {
            authorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
            contentType.set(exchange.getRequestHeaders().getFirst("Content-Type"));
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            json(exchange, 200,
                    "{\"name\":\"reference.png\",\"subfolder\":\"ai-fusion-video\",\"type\":\"input\"}");
        });

        ComfyUiUploadResult result = client.uploadImage(
                apiConfig, "png-data".getBytes(StandardCharsets.UTF_8),
                "reference.png", "image/png", "ai-fusion-video");

        assertThat(result.workflowValue()).isEqualTo("ai-fusion-video/reference.png");
        assertThat(authorization.get()).isEqualTo("Bearer test-token");
        assertThat(contentType.get()).startsWith("multipart/form-data; boundary=");
        assertThat(requestBody.get())
                .contains("name=\"image\"; filename=\"reference.png\"")
                .contains("name=\"type\"")
                .contains("input")
                .contains("name=\"subfolder\"")
                .contains("ai-fusion-video");
    }

    @Test
    void uploadImageRejectsResponseMissingOfficialSubfolderField() {
        server.createContext("/upload/image", exchange -> json(exchange, 200,
                "{\"name\":\"reference.png\",\"type\":\"input\"}"));

        assertThatThrownBy(() -> client.uploadImage(
                apiConfig, "png-data".getBytes(StandardCharsets.UTF_8),
                "reference.png", "image/png", ""))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("subfolder");
    }

    @Test
    void getJobParsesOfficialCompletedOutputs() {
        String promptId = UUID.randomUUID().toString();
        server.createContext("/api/jobs/" + promptId, exchange -> json(exchange, 200,
                "{\"id\":\"" + promptId + "\",\"status\":\"completed\","
                        + "\"outputs\":{\"9\":{\"images\":[{\"filename\":\"result.png\","
                        + "\"subfolder\":\"\",\"type\":\"output\"}]}},\"execution_error\":null}"));

        ComfyUiJobResult result = client.getJob(apiConfig, promptId);

        assertThat(result.completed()).isTrue();
        assertThat(result.outputs().path("9").path("images").get(0).path("filename").asText())
                .isEqualTo("result.png");
    }

    @Test
    void getJobRejectsStatusOutsideOfficialStateSet() {
        String promptId = UUID.randomUUID().toString();
        server.createContext("/api/jobs/" + promptId, exchange -> json(exchange, 200,
                "{\"id\":\"" + promptId + "\",\"status\":\"done\",\"outputs\":{}}"));

        assertThatThrownBy(() -> client.getJob(apiConfig, promptId))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("未知状态");
    }

    @Test
    void cancelJobParsesOfficialBooleanResponse() {
        String promptId = UUID.randomUUID().toString();
        server.createContext("/api/jobs/" + promptId + "/cancel", exchange ->
                json(exchange, 200, "{\"cancelled\":false}"));

        assertThat(client.cancelJob(apiConfig, promptId)).isFalse();
    }

    @Test
    void downloadOutputUsesOfficialViewDescriptorAndBearerToken() throws IOException {
        AtomicReference<String> authorization = new AtomicReference<>();
        AtomicReference<String> rawQuery = new AtomicReference<>();
        byte[] media = "video-data".getBytes(StandardCharsets.UTF_8);
        server.createContext("/view", exchange -> {
            authorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
            rawQuery.set(exchange.getRequestURI().getRawQuery());
            binary(exchange, "video/mp4", media);
        });

        try (ComfyUiDownloadedFile result = client.downloadOutput(
                apiConfig, "clip one.mp4", "folder/sub", "output")) {
            assertThat(Files.readAllBytes(result.path())).isEqualTo(media);
            assertThat(result.contentType()).isEqualTo("video/mp4");
            assertThat(result.extension()).isEqualTo("mp4");
            assertThat(result.size()).isEqualTo(media.length);
        }
        assertThat(authorization.get()).isEqualTo("Bearer test-token");
        assertThat(rawQuery.get()).isEqualTo(
                "filename=clip+one.mp4&subfolder=folder%2Fsub&type=output");
    }

    @Test
    void copyBoundedStopsBeforeWritingPastConfiguredLimit() throws IOException {
        Path target = Files.createTempFile("comfyui-client-test-", ".bin");
        try {
            byte[] body = "123456789".getBytes(StandardCharsets.UTF_8);

            assertThatThrownBy(() -> ComfyUiNativeClient.copyBounded(
                    new ByteArrayInputStream(body), target, 8))
                    .isInstanceOf(BusinessException.class)
                    .hasMessageContaining("超过 1GB 限制");
            assertThat(Files.size(target)).isLessThanOrEqualTo(8L);
        } finally {
            Files.deleteIfExists(target);
        }
    }

    @Test
    void readJsonBoundedStopsBeforeBufferingPastConfiguredLimit() {
        byte[] body = "123456789".getBytes(StandardCharsets.UTF_8);

        assertThatThrownBy(() -> ComfyUiNativeClient.readJsonBounded(
                new ByteArrayInputStream(body), 8))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("超过 16MB 限制");
    }

    private void json(HttpExchange exchange, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.close();
    }

    private void binary(HttpExchange exchange, String contentType, byte[] body) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", contentType);
        exchange.sendResponseHeaders(200, body.length);
        exchange.getResponseBody().write(body);
        exchange.close();
    }
}
