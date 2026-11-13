package com.stonewu.fusion.service.ai.provider;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.agentscope.kernel.OwnedChatModel;
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

class YoukouTextProtocolTests {
    @Test
    void reasoningModelSendsOnlyThePublishedReasoningEffortParameter() throws Exception {
        var body = new AtomicReference<JsonNode>();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/chat/completions", exchange -> {
            body.set(new ObjectMapper().readTree(exchange.getRequestBody()));
            byte[] response = (StreamingProviderFixtures.first("openai") + StreamingProviderFixtures.last("openai"))
                    .getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
            exchange.sendResponseHeaders(200, response.length);
            exchange.getResponseBody().write(response);
            exchange.close();
        });
        server.start();
        try {
            String url = "http://127.0.0.1:" + server.getAddress().getPort();
            var api = ApiConfig.builder().apiKey("test-key").apiUrl(url).autoAppendV1Path(true).build();
            var context = AiProviderContext.builder().apiConfig(api).apiKey("test-key").baseUrl(url)
                    .platform("openai_compatible").modelName("DeepSeek V4 Pro").singleRequest(true)
                    .model(AiModel.builder().supportReasoning(true).build())
                    .config(Map.of("sendIncludeReasoning", false, "reasoningEffort", "high")).build();
            try (var model = OwnedChatModel.owned(new OpenAiCompatibleAiProvider().createAgentScopeModel(context))) {
                var messages = List.of(Msg.builder().role(MsgRole.USER)
                        .content(TextBlock.builder().text("写一句场景描述").build()).build());
                var response = model.model().stream(messages, List.of(), null).collectList().block(Duration.ofSeconds(5));
                assertThat(response).isNotEmpty();
            }
            assertThat(body.get().path("reasoning_effort").asText()).isEqualTo("high");
            assertThat(body.get().has("include_reasoning")).isFalse();
            assertThat(body.get().path("stream").asBoolean()).isTrue();
        } finally {
            server.stop(0);
        }
    }
}
