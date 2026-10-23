package com.stonewu.fusion.service.generation.transport;

import com.alibaba.dashscope.aigc.imagegeneration.ImageGenerationMessage;
import com.alibaba.dashscope.aigc.imagegeneration.ImageGenerationParam;
import com.alibaba.dashscope.aigc.videosynthesis.VideoSynthesisParam;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.dashscope.DashScopeGenerationSupport;
import com.stonewu.fusion.service.ai.dashscope.DashScopeSingleRequestClient;
import com.stonewu.fusion.service.ai.googleflow.GoogleFlowReverseApiClient;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.dashscope.DashScopeImageStrategy;
import com.stonewu.fusion.service.generation.image.strategy.gemini.VertexAiImageStrategy;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.media.GenerationMediaService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import com.volcengine.ark.runtime.model.images.generation.GenerateImagesRequest;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;

class GenerationSingleRequestTests {
    private HttpServer server;
    private ApiConfig config;
    private String base;
    private int responseCode;
    private String responseBody;
    private final List<Received> received = new CopyOnWriteArrayList<>();

    @BeforeEach
    void start() throws IOException {
        responseCode = 503;
        responseBody = "{\"code\":\"ServiceUnavailable\",\"message\":\"uncertain\"}";
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", this::handle);
        server.start();
        base = "http://127.0.0.1:" + server.getAddress().getPort();
        config = ApiConfig.builder().apiUrl(base).apiKey("test-api-key").build();
    }

    @AfterEach void stop() { server.stop(0); }

    private void handle(HttpExchange exchange) throws IOException {
        received.add(new Received(exchange.getRequestMethod(), exchange.getRequestURI().getRawPath(),
                new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8),
                exchange.getRequestHeaders().getFirst("Authorization"),
                exchange.getRequestHeaders().getFirst("Proxy-Authorization"),
                exchange.getRequestHeaders().getFirst("X-DashScope-Async")));
        if (responseCode == 0) { exchange.close(); return; }
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.getResponseHeaders().add("Retry-After", "0");
        exchange.getResponseHeaders().add("Location", base + "/redirected");
        byte[] body = responseBody.getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(responseCode, body.length);
        exchange.getResponseBody().write(body);
        exchange.close();
    }

    private OkHttpClient client() {
        return GenerationHttpClients.singleRequestBuilder().callTimeout(5, TimeUnit.SECONDS).build();
    }

    private Request paidRequest() {
        return new Request.Builder().url(base + "/paid")
                .post(RequestBody.create("{\"prompt\":\"test\"}", MediaType.get("application/json"))).build();
    }

    @ParameterizedTest
    @ValueSource(ints = {503, 408, 429, 307, 308})
    void responseCannotAutomaticallyResendPaidPost(int code) throws IOException {
        responseCode = code;
        try (var response = client().newCall(paidRequest()).execute()) {
            assertThat(response.code()).isEqualTo(code);
        }
        assertThat(received).hasSize(1);
        assertThat(received.getFirst().path()).isEqualTo("/paid");
    }

    @Test void missingResponseCannotAutomaticallyResendPaidPost() {
        responseCode = 0;
        assertThatThrownBy(() -> client().newCall(paidRequest()).execute()).isInstanceOf(IOException.class);
        assertThat(received).hasSize(1);
    }

    @Test void hiddenSdkInterceptorCannotSendSameCallTwice() {
        var sdkClient = client().newBuilder().addInterceptor(chain -> {
            chain.proceed(chain.request()).close();
            return chain.proceed(chain.request());
        }).build();
        assertThatThrownBy(() -> sdkClient.newCall(paidRequest()).execute())
                .isInstanceOf(IOException.class).hasMessageContaining("禁止自动重复发送");
        assertThat(received).hasSize(1);
    }

    @Test void oneQueryCannotRunTwoNetworkExchanges() {
        assertThatThrownBy(() -> client().newCall(new Request.Builder().url(base + "/state").build()).execute())
                .isInstanceOf(IOException.class).hasMessageContaining("禁止自动重复发送");
        assertThat(received).hasSize(1);
    }

    @Test void springClientKeepsPayloadAndDoesNotResendServerError() {
        var spring = GenerationHttpClients.restClientBuilder(config).build();
        assertThatThrownBy(() -> spring.post().uri(base + "/spring").body(Map.of("prompt", "original"))
                .retrieve().body(String.class)).hasMessageContaining("503")
                .satisfies(failure -> assertThat(GenerationProviderFailure.find(failure)).isNotNull());
        assertThat(received).hasSize(1);
        assertThat(received.getFirst().body()).contains("\"prompt\":\"original\"");
    }

    @Test void arkClientKeepsSdkAuthenticationAndNeverUsesSdkServerErrorRetry() {
        config.setApiUrl(base + "/api/v3");
        var ark = ArkGenerationClients.create(config);
        try {
            assertThatThrownBy(() -> ark.generateImages(GenerateImagesRequest.builder()
                    .model("seedream-test").prompt("original").size("1024x1024").build()))
                    .isInstanceOf(RuntimeException.class)
                    .satisfies(failure -> assertThat(GenerationProviderFailure.find(failure)).isNotNull());
            assertThat(received).hasSize(1);
            assertThat(received.getFirst().path()).isEqualTo("/api/v3/images/generations");
            assertThat(received.getFirst().authorization()).isEqualTo("Bearer test-api-key");
            assertThat(received.getFirst().body()).contains("original", "seedream-test");
        } finally { ark.shutdownExecutor(); }
    }

    @Test void dashScopeOfficialClientKeepsSdkImageProtocolWithoutRetries() {
        var dash = new DashScopeSingleRequestClient(config);
        var param = ImageGenerationParam.builder().apiKey(config.getApiKey()).model("qwen-image")
                .messages(List.of(ImageGenerationMessage.builder().role("user")
                        .content(List.of(Map.of("text", "original"))).build())).build();
        var prepared = dash.prepareImage(param, false);
        assertThat(received).isEmpty();
        assertThatThrownBy(prepared::get).isInstanceOf(Exception.class)
                .satisfies(failure -> assertThat(GenerationProviderFailure.find(failure)).isNotNull());
        assertThat(received).hasSize(1);
        assertThat(received.getFirst().path()).isEqualTo("/api/v1/services/aigc/multimodal-generation/generation");
        assertThat(received.getFirst().authorization()).isEqualTo("Bearer test-api-key");
        assertThat(received.getFirst().body()).contains("\"messages\"", "original");
    }

    @Test void dashScopeVideoSubmitAndEachExplicitQueryUseSeparatePermits() throws Exception {
        responseCode = 200;
        responseBody = "{\"request_id\":\"request-1\",\"output\":{\"task_id\":\"task-1\",\"task_status\":\"PENDING\"}}";
        var dash = new DashScopeSingleRequestClient(config);
        var prepared = dash.prepareVideo(VideoSynthesisParam.builder().apiKey(config.getApiKey())
                .model("wan2.6-t2v").prompt("original").build());
        assertThat(received).isEmpty();
        var result = prepared.get();
        assertThat(result.getOutput().getTaskId()).isEqualTo("task-1");
        dash.fetch("task-1", config.getApiKey());
        dash.fetch("task-1", config.getApiKey());
        assertThat(received).hasSize(3);
        assertThat(received.getFirst().path()).isEqualTo("/api/v1/services/aigc/video-generation/video-synthesis");
        assertThat(received.getFirst().async()).isEqualTo("enable");
        assertThat(received.subList(1, 3)).allMatch(request -> request.method().equals("GET") && request.path().equals("/api/v1/tasks/task-1"));
    }

    @Test void dashScopeQueryDoesNotReopenOriginalInputMedia() {
        responseCode = 200;
        responseBody = "{\"output\":{\"task_id\":\"original\",\"task_status\":\"PENDING\"}}";
        var media = mock(DashScopeGenerationSupport.class);
        var strategy = new DashScopeImageStrategy(mock(ImageGenerationService.class), mock(AiModelService.class), media);
        var task = ImageTask.builder().width(1024).height(1024).count(1).refImageUrls("[\"/media/no-longer-present.png\"]").build();
        var result = strategy.query("original", task, AiModel.builder().code("wanx2.1-t2i-turbo").build(), config);
        assertThat(result.state()).isEqualTo(RemoteGenerationResult.State.PENDING);
        verifyNoInteractions(media);
        assertThat(received).hasSize(1).allMatch(value -> value.method().equals("GET"));
    }

    @Test void googleFlowReferenceFailureHappensDuringPreparationWithoutGenerationPost() {
        responseCode = 404;
        var flow = new GoogleFlowReverseApiClient(mock(StorageConfigService.class), mock(PresetArtStyleResourceResolver.class));
        assertThatThrownBy(() -> flow.prepare("original", "image-model", List.of(base + "/missing.png"), config))
                .isInstanceOf(RuntimeException.class);
        assertThat(received).hasSize(1).allMatch(value -> value.method().equals("GET") && value.authorization() == null);
    }

    @Test void vertexCredentialFailureIsDetectedBeforeARequestCanBeSubmitted() {
        config.setAppId("test-project");
        config.setAppSecret("{}");
        config.setApiUrl(base + "/v1/projects/test/locations/us-central1/publishers/google/models/imagen:predict");
        var vertex = new VertexAiImageStrategy(mock(ImageGenerationService.class), mock(AiModelService.class), mock(GenerationMediaService.class));
        assertThatThrownBy(() -> vertex.prepare(ImageTask.builder().prompt("original").count(1).build(),
                AiModel.builder().code("imagen-4.0-generate-001").build(), config)).hasMessageContaining("身份准备失败");
        assertThat(received).isEmpty();
    }

    @Test void configuredHttpProxyAuthenticatesFirstExchangeWithoutReplayingBody() throws IOException {
        responseCode = 200;
        responseBody = "{}";
        config.setProxyType("http");
        config.setProxyHost("127.0.0.1");
        config.setProxyPort(server.getAddress().getPort());
        config.setProxyUsername("proxy-user");
        config.setProxyPassword("proxy-pass");
        var proxied = AiProxySupport.okHttpClient(client(), config);
        var request = paidRequest().newBuilder().url("http://provider.invalid/paid").build();
        try (var response = proxied.newCall(request).execute()) { assertThat(response.code()).isEqualTo(200); }
        assertThat(received).hasSize(1);
        assertThat(received.getFirst().proxyAuthorization()).isEqualTo("Basic cHJveHktdXNlcjpwcm94eS1wYXNz");
        assertThat(received.getFirst().authorization()).isNull();
    }

    private record Received(String method, String path, String body, String authorization, String proxyAuthorization, String async) {}
}
