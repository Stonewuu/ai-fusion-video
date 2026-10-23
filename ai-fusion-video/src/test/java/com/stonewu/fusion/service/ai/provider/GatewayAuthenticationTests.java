package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.agentscope.kernel.OwnedChatModel;
import com.stonewu.fusion.service.ai.ChatModelFactory;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.sun.net.httpserver.HttpServer;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** 保护兼容网关的真实地址及 Bearer 鉴权，避免 SDK 把请求发回原生厂商。 */
class GatewayAuthenticationTests {
    @ParameterizedTest
    @ValueSource(strings = {"anthropic", "gemini", "responses"})
    void streamedNativeProtocolsHonorTheGateway(String protocol) throws Exception {
        try (var endpoint = new Endpoint(protocol)) {
            var context = context(protocol, true, endpoint.url());
            var provider = provider(protocol);
            try (var owned = OwnedChatModel.owned(provider.createAgentScopeModel(context))) {
                var input = Msg.builder().role(MsgRole.USER).name("user")
                        .content(TextBlock.builder().text("描述场景").build()).build();
                var output = owned.model().stream(List.of(input), List.of(), null).collectList()
                        .block(Duration.ofSeconds(8));
                assertThat(output).isNotEmpty();
                endpoint.assertContract();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"anthropic", "gemini", "responses"})
    void connectivityChecksUseTheSameGatewayAsGeneration(String protocol) throws Exception {
        try (var endpoint = new Endpoint(protocol)) {
            var context = context(protocol, false, endpoint.url());
            var contextFactory = mock(AiProviderContextFactory.class);
            when(contextFactory.createForModel(context.getModel())).thenReturn(context);
            var providers = new AiProviderService(contextFactory, new AiProviderRegistry(List.of(provider(protocol))),
                    mock(AiModelMetadataResolver.class), new ModelPresetService());
            var factory = new ChatModelFactory(providers);
            assertThat(factory.checkConnectivity(context.getModel(), "描述场景")).isEqualTo("雨落古城。");
            endpoint.assertContract();
        }
    }

    private AiProviderContext context(String protocol, boolean single, String url) {
        return AiProviderContext.builder().apiConfig(ApiConfig.builder().apiUrl(url).apiKey("gateway-test-key").build())
                .apiKey("gateway-test-key").baseUrl(url + "/v1")
                .platform(protocol.equals("responses") ? "openai_compatible" : protocol)
                .model(AiModel.builder().id(1L).code("native-model").modelType(1).build()).modelName("native-model")
                .singleRequest(single).config(Map.of("gatewayAuthMode", "bearer", "maxTokens", 512,
                        "apiMode", protocol.equals("responses") ? "responses" : "chat_completions")).build();
    }

    private AiProvider provider(String protocol) {
        return switch (protocol) {
            case "anthropic" -> new AnthropicAiProvider();
            case "gemini" -> new GeminiAiProvider();
            case "responses" -> new OpenAiCompatibleAiProvider();
            default -> throw new IllegalArgumentException(protocol);
        };
    }

    private static final class Endpoint implements AutoCloseable {
        private final HttpServer server;
        private final String protocol;
        private final List<Observed> observed = new CopyOnWriteArrayList<>();

        Endpoint(String protocol) throws Exception {
            this.protocol = protocol;
            server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            server.createContext("/gateway/", exchange -> {
                observed.add(new Observed(exchange.getRequestURI().toString(),
                        exchange.getRequestHeaders().getFirst("Authorization"),
                        new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)));
                String response = StreamingProviderFixtures.first(protocol) + StreamingProviderFixtures.last(protocol);
                byte[] bytes = response.getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
                exchange.sendResponseHeaders(200, bytes.length);
                exchange.getResponseBody().write(bytes);
                exchange.close();
            });
            server.start();
        }

        String url() {
            return "http://127.0.0.1:" + server.getAddress().getPort() + "/gateway";
        }

        void assertContract() {
            assertThat(observed).hasSize(1);
            assertThat(observed.getFirst().authorization()).isEqualTo("Bearer gateway-test-key");
            assertThat(observed.getFirst().path()).startsWith(switch (protocol) {
                case "anthropic" -> "/gateway/v1/messages";
                case "gemini" -> "/gateway/v1beta/models/native-model:";
                case "responses" -> "/gateway/v1/responses";
                default -> throw new IllegalArgumentException(protocol);
            });
            assertThat(observed.getFirst().body()).doesNotContain("gateway-test-key");
        }

        @Override
        public void close() {
            server.stop(0);
        }
    }

    private record Observed(String path, String authorization, String body) {
    }
}
