package com.stonewu.fusion.service.ai.apixo;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.agentscope.kernel.OwnedChatModel;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.ai.provider.AiProviderContextFactory;
import com.stonewu.fusion.service.ai.provider.AnthropicAiProvider;
import com.stonewu.fusion.service.ai.provider.GeminiAiProvider;
import com.stonewu.fusion.service.ai.provider.OpenAiCompatibleAiProvider;
import com.stonewu.fusion.service.ai.provider.StreamingProviderFixtures;
import com.sun.net.httpserver.HttpServer;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** 三类对话必须保持各自端点和鉴权，不把 Responses 发往 Chat Completions。 */
class ApixoTextProtocolTests {
    record Call(String path, String authorization, String body) {
    }

    @ParameterizedTest
    @CsvSource({
            "anthropic,claude-haiku-4-5-20251001,anthropic,x-api-key,/v1/messages",
            "openai_responses,gpt-5-mini,responses,Authorization,/v1/responses",
            "gemini,gemini-2.5-flash-lite,gemini,x-goog-api-key,/v1beta/models/gemini-2.5-flash-lite:streamGenerateContent"
    })
    void allChatFamiliesUseTheirDocumentedProtocol(String protocol, String name, String fixture,
                                                   String keyHeader, String expectedPath) throws Exception {
        var calls = new CopyOnWriteArrayList<Call>();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            calls.add(new Call(exchange.getRequestURI().getPath(), exchange.getRequestHeaders().getFirst(keyHeader),
                    new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)));
            byte[] response = (StreamingProviderFixtures.first(fixture) + StreamingProviderFixtures.last(fixture))
                    .getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
            exchange.sendResponseHeaders(200, response.length);
            exchange.getResponseBody().write(response);
            exchange.close();
        });
        server.start();
        try {
            var config = ApiConfig.builder().id(3L).providerId("apixo").platform("openai_compatible")
                    .apiUrl("http://127.0.0.1:" + server.getAddress().getPort()).apiKey("fixture-api-key")
                    .autoAppendV1Path(false).status(1).build();
            var configs = mock(ApiConfigService.class);
            when(configs.getById(3L)).thenReturn(config);
            var factory = new AiProviderContextFactory(configs, new AiModelMetadataResolver(configs));
            var model = AiModel.builder().id(4L).apiConfigId(3L).modelType(1).code(name)
                    .modelProtocol(protocol).supportReasoning(false).config("{}").build();
            var context = factory.createForModel(model).toBuilder().singleRequest(true).build();
            var client = switch (protocol) {
                case "anthropic" -> new AnthropicAiProvider().createAgentScopeModel(context);
                case "gemini" -> new GeminiAiProvider().createAgentScopeModel(context);
                default -> new OpenAiCompatibleAiProvider().createAgentScopeModel(context);
            };
            try (var owned = OwnedChatModel.owned(client)) {
                var message = Msg.builder().name("user").role(MsgRole.USER)
                        .content(TextBlock.builder().text("描述一处场景").build()).build();
                var chunks = owned.model().stream(List.of(message), List.of(), null)
                        .concatMapIterable(chunk -> chunk.getContent()).ofType(TextBlock.class)
                        .map(TextBlock::getText).collectList().block(Duration.ofSeconds(10));
                assertThat(String.join("", chunks)).isEqualTo("雨落古城。");
            }
            assertThat(calls).singleElement().satisfies(call -> {
                assertThat(call.path()).isEqualTo(expectedPath);
                assertThat(call.authorization()).isEqualTo(keyHeader.equals("Authorization")
                        ? "Bearer fixture-api-key" : "fixture-api-key");
                assertThat(call.body()).doesNotContain("fixture-api-key");
            });
        } finally {
            server.stop(0);
        }
    }

    @Test
    void officialGenerationBaseRoutesTextToTheSeparateGatewayWithoutMutatingTheConnection() {
        var api = ApiConfig.builder().apiUrl("https://api.apixo.ai/api/v1").build();
        assertThat(ApixoEndpoints.chatBase(api)).isEqualTo("https://llm.apixo.ai/v1");
        assertThat(ApixoEndpoints.submit(api, "seedream-5-5"))
                .isEqualTo("https://api.apixo.ai/api/v1/generateTask/seedream-5-5");
        assertThat(api.getApiUrl()).isEqualTo("https://api.apixo.ai/api/v1");
    }
}
