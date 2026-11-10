package com.stonewu.fusion.service.ai.provider;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.sun.net.httpserver.HttpServer;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;

/** Gemini 4 拒绝 thinkingBudget；用出站请求验证思考等级不会被旧预算参数覆盖。 */
class GoogleThinkingOptionsTests {
    @Test
    void sendsModernThinkingLevelWithoutLegacyBudget() throws Exception {
        var body = new AtomicReference<String>();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            body.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            byte[] response = (StreamingProviderFixtures.first("gemini") + StreamingProviderFixtures.last("gemini"))
                    .getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
            exchange.sendResponseHeaders(200, response.length);
            try (var output = exchange.getResponseBody()) {
                output.write(response);
            }
        });
        server.start();
        var context = AiProviderContext.builder().modelName("gemini-4-argon").apiKey("fixture-key")
                .baseUrl("http://127.0.0.1:" + server.getAddress().getPort()).singleRequest(true)
                .model(AiModel.builder().supportReasoning(true).build())
                .apiConfig(ApiConfig.builder().platform("gemini").build())
                .config(Map.of("reasoningEffort", "high")).build();
        try (var model = (SingleRequestGeminiModel) new GeminiAiProvider().createAgentScopeModel(context)) {
            var messages = List.of(Msg.builder().name("user").role(MsgRole.USER)
                    .content(TextBlock.builder().text("测试").build()).build());
            model.stream(messages, List.of(), null).collectList().block(Duration.ofSeconds(5));
            var thinking = new ObjectMapper().readTree(body.get()).at("/generationConfig/thinkingConfig");
            assertThat(thinking.path("thinkingBudget").isMissingNode()).isTrue();
            assertThat(thinking.path("thinkingLevel").asText()).isEqualTo("HIGH");
        } finally {
            server.stop(0);
        }
    }
}
