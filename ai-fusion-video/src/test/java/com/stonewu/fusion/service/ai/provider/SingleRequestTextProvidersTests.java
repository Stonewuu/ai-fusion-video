package com.stonewu.fusion.service.ai.provider;

import com.google.auth.oauth2.AccessToken;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.genai.GenerationHttpApiClient;
import com.google.genai.types.HttpOptions;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.agentscope.kernel.OwnedChatModel;
import com.stonewu.fusion.service.ai.agentscope.message.AgentModelHttpFailure;
import com.sun.net.httpserver.HttpServer;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.model.ChatModelBase;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import reactor.test.StepVerifier;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 真实 HTTP 验证 SDK 与底层传输均不能隐藏重发，不访问外部模型服务。 */
class SingleRequestTextProvidersTests {
    private static final List<String> PROTOCOLS = List.of("openai", "responses", "anthropic", "gemini", "vertex", "dashscope", "ollama");
    private static final List<Msg> MESSAGES = List.of(Msg.builder().name("user").role(MsgRole.USER)
            .content(TextBlock.builder().text("生成一句场景描述").build()).build());

    static Stream<Object[]> failures() {
        return PROTOCOLS.stream().flatMap(protocol -> Stream.of(307, 408, 429, 503).map(status -> new Object[]{protocol, status}));
    }

    @ParameterizedTest @MethodSource("failures")
    void providerErrorsAndRedirectsNeverCauseAnotherPaidPost(String protocol, int status) throws Exception {
        try (var endpoint = new Endpoint(protocol, status);
             var model = OwnedChatModel.owned(model(protocol, endpoint.url()))) {
            assertThatThrownBy(() -> model.model().stream(MESSAGES, List.of(), null).collectList().block(Duration.ofSeconds(8)))
                    .isInstanceOf(RuntimeException.class)
                    .satisfies(failure -> assertThat(AgentModelHttpFailure.status(failure)).isEqualTo(status));
            assertThat(endpoint.requests).hasValue(1);
        }
    }

    @ParameterizedTest @ValueSource(strings = {"openai", "responses", "anthropic", "gemini", "vertex", "dashscope", "ollama"})
    void officialFormattersStillReadTheCompleteTextResponse(String protocol) throws Exception {
        try (var endpoint = new Endpoint(protocol, 200);
             var model = OwnedChatModel.owned(model(protocol, endpoint.url()))) {
            var response = model.model().stream(MESSAGES, List.of(), null).collectList().block(Duration.ofSeconds(8));
            assertThat(response).isNotEmpty();
            String text = response.stream().flatMap(chunk -> chunk.getContent().stream()).filter(TextBlock.class::isInstance)
                    .map(TextBlock.class::cast).map(TextBlock::getText).reduce("", String::concat);
            assertThat(text).isEqualTo("雨落古城。");
            assertThat(response).anySatisfy(chunk -> {
                assertThat(chunk.getUsage()).isNotNull();
                assertThat(chunk.getUsage().getInputTokens()).isEqualTo(10);
                assertThat(chunk.getUsage().getOutputTokens()).isEqualTo(5);
            });
            assertThat(response).anySatisfy(chunk -> assertThat(chunk.getFinishReason()).isNotBlank());
            assertThat(endpoint.requests).hasValue(1);
            assertThatThrownBy(() -> model.model().stream(MESSAGES, List.of(), null).collectList().block(Duration.ofSeconds(8)))
                    .isInstanceOf(RuntimeException.class);
            assertThat(endpoint.requests).hasValue(1);
        }
    }

    @ParameterizedTest @ValueSource(strings = {"openai", "responses", "anthropic", "gemini", "vertex", "dashscope", "ollama"})
    void firstTextArrivesWhileTheProviderIsStillWaitingToSendTheRest(String protocol) throws Exception {
        try (var endpoint = new Endpoint(protocol, 200, true);
             var model = OwnedChatModel.owned(model(protocol, endpoint.url()))) {
            StepVerifier.create(model.model().stream(MESSAGES, List.of(), null)
                            .concatMapIterable(chunk -> chunk.getContent()).ofType(TextBlock.class).map(TextBlock::getText))
                    .expectNext("雨落")
                    .then(() -> endpoint.release.countDown())
                    .expectNext("古城。")
                    .expectComplete().verify(Duration.ofSeconds(5));
            assertThat(endpoint.requests).hasValue(1);
        }
    }

    @ParameterizedTest @ValueSource(strings = {"openai", "responses", "anthropic", "gemini", "vertex", "dashscope", "ollama"})
    void cancellationDoesNotWaitForTheRemainingProviderOutput(String protocol) throws Exception {
        try (var endpoint = new Endpoint(protocol, 200, true);
             var model = OwnedChatModel.owned(model(protocol, endpoint.url()))) {
            StepVerifier.create(model.model().stream(MESSAGES, List.of(), null)
                            .concatMapIterable(chunk -> chunk.getContent()).ofType(TextBlock.class).map(TextBlock::getText))
                    .expectNext("雨落").thenCancel().verify(Duration.ofSeconds(5));
            assertThat(endpoint.release.getCount()).isEqualTo(1);
            assertThat(endpoint.requests).hasValue(1);
        }
    }

    private ChatModelBase model(String protocol, String url) {
        var api = ApiConfig.builder().id(1L).apiKey("local-test-key").apiUrl(url).autoAppendV1Path(true).build();
        if (protocol.equals("vertex")) {
            var credentials = GoogleCredentials.create(new AccessToken("local-test-token", Date.from(Instant.now().plusSeconds(600))));
            return new SingleRequestGeminiModel(new GenerationHttpApiClient("test-project", "us-central1", credentials,
                    HttpOptions.builder().baseUrl(url).build(), api), "gemini-test", null);
        }
        var context = AiProviderContext.builder().apiConfig(api).apiKey("local-test-key").baseUrl(url)
                .model(AiModel.builder().modelType(1).supportReasoning(false).build()).modelName("test-text-model")
                .platform(protocol.equals("responses") ? "openai_compatible" : protocol)
                .config(protocol.equals("responses") ? Map.of("apiMode", "responses") : Map.of())
                .singleRequest(true).build();
        return switch (protocol) {
            case "openai", "responses" -> new OpenAiCompatibleAiProvider().createAgentScopeModel(context);
            case "anthropic" -> new AnthropicAiProvider().createAgentScopeModel(context);
            case "gemini" -> new GeminiAiProvider().createAgentScopeModel(context);
            case "dashscope" -> new DashScopeAiProvider().createAgentScopeModel(context);
            case "ollama" -> new OllamaAiProvider().createAgentScopeModel(context);
            default -> throw new IllegalArgumentException(protocol);
        };
    }

    private static final class Endpoint implements AutoCloseable {
        private final HttpServer server;
        private final AtomicInteger requests = new AtomicInteger();
        private final CountDownLatch release;
        Endpoint(String protocol, int status) throws IOException {
            this(protocol, status, false);
        }
        Endpoint(String protocol, int status, boolean gated) throws IOException {
            release = new CountDownLatch(gated ? 1 : 0);
            server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            server.createContext("/", exchange -> {
                requests.incrementAndGet();
                exchange.getRequestBody().readAllBytes();
                int code = requests.get() == 1 ? status : 200;
                String body = code == 200 ? StreamingProviderFixtures.first(protocol) : "{\"error\":{\"message\":\"test-error\",\"type\":\"server_error\"}}";
                var bytes = body.getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", code == 200 && !protocol.equals("ollama") ? "text/event-stream" : "application/json");
                exchange.getResponseHeaders().set("Retry-After", "0");
                if (code == 307) exchange.getResponseHeaders().set("Location", url() + "/redirect-target");
                exchange.sendResponseHeaders(code, 0);
                try (var output = exchange.getResponseBody()) {
                    output.write(bytes);
                    output.flush();
                    if (code == 200) {
                        try {
                            if (!release.await(6, TimeUnit.SECONDS)) {
                                throw new IOException("首段未及时到达消费者");
                            }
                        } catch (InterruptedException interrupted) {
                            Thread.currentThread().interrupt();
                            throw new IOException(interrupted);
                        }
                        output.write(StreamingProviderFixtures.last(protocol).getBytes(StandardCharsets.UTF_8));
                    }
                }
            });
            server.start();
        }
        String url() { return "http://127.0.0.1:" + server.getAddress().getPort(); }
        @Override public void close() { release.countDown(); server.stop(0); }
    }

}
