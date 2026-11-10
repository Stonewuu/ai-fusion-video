package com.stonewu.fusion.service.ai.provider;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.agentscope.message.AgentModelCallFailure;
import com.stonewu.fusion.service.ai.agentscope.message.AgentModelHttpFailure;
import com.sun.net.httpserver.HttpServer;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.message.UserMessage;
import io.agentscope.core.model.ToolSchema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 使用本地 HTTP 流验证 Responses 工具声明和交错参数，防止正文测试通过而智能体工具不可用。 */
class OpenAiResponsesToolsTests {
    private final ObjectMapper json = new ObjectMapper();
    private final ToolSchema query = ToolSchema.builder().name("get_project").description("查询项目")
            .parameters(Map.of("type", "object", "properties", Map.of("projectId", Map.of("type", "integer")),
                    "required", List.of("projectId"))).build();

    @Test
    void unsetStrictAndInterleavedCallsKeepCompleteArgumentsAndCallIds() throws Exception {
        var body = new AtomicReference<JsonNode>();
        var server = endpoint(body, events(false));
        try {
            var model = model(server);
            var output = model.stream(List.of(new UserMessage("查询两个项目")), List.of(query), null)
                    .collectList().block(Duration.ofSeconds(10));
            var calls = output.stream().flatMap(chunk -> chunk.getContent().stream()).filter(ToolUseBlock.class::isInstance)
                    .map(ToolUseBlock.class::cast).toList();
            assertThat(body.get().path("tools").get(0).path("strict").isBoolean()).isTrue();
            assertThat(body.get().path("tools").get(0).path("strict").booleanValue()).isFalse();
            assertThat(calls).extracting(ToolUseBlock::getId).containsExactly("call-b", "call-a");
            assertThat(calls).extracting(ToolUseBlock::getName).containsExactly("get_project", "get_project");
            assertThat(calls.get(0).getInput()).containsEntry("projectId", 8);
            assertThat(calls.get(1).getInput()).containsEntry("projectId", 4);
            assertThat(output).anySatisfy(chunk -> assertThat(chunk.getFinishReason()).isEqualTo("tool_calls"));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void invalidArgumentsAreRejectedInsteadOfCallingTheToolWithInventedRawParameter() throws Exception {
        var server = endpoint(new AtomicReference<>(), events(true));
        try {
            assertThatThrownBy(() -> model(server).stream(List.of(new UserMessage("查询项目")), List.of(query), null)
                    .collectList().block(Duration.ofSeconds(10)))
                    .isInstanceOf(IllegalArgumentException.class).hasMessageContaining("JSON 对象");
        } finally {
            server.stop(0);
        }
    }

    /** 使用 SDK 解析本地 HTTP 响应，避免只用自造异常验证而漏掉生产异常类型。 */
    @ParameterizedTest
    @CsvSource({"520,server_error,true", "409,conflict,true", "429,rate_limit_exceeded,true", "429,insufficient_quota,false",
            "401,invalid_api_key,false", "403,permission_denied,false", "422,invalid_request_error,false"})
    void sdkHttpErrorsRetainStatusAndRetryabilityWithoutHiddenNetworkRetries(int status, String code, boolean retryable) throws Exception {
        var attempts = new AtomicInteger();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            attempts.incrementAndGet();
            exchange.getRequestBody().readAllBytes();
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.getResponseHeaders().set("Retry-After", "30");
            String message = code.equals("rate_limit_exceeded")
                    ? "Rate limit reached. Please try again shortly. See https://platform.openai.com/account/billing."
                    : "request rejected";
            byte[] body = json.writeValueAsBytes(Map.of("error", Map.of("code", code, "message", message, "type", code)));
            exchange.sendResponseHeaders(status, body.length);
            try (var output = exchange.getResponseBody()) {
                output.write(body);
            }
        });
        server.start();
        try {
            assertThatThrownBy(() -> model(server).stream(List.of(new UserMessage("test")), List.of(), null)
                    .collectList().block(Duration.ofSeconds(10)))
                    .satisfies(error -> {
                        assertThat(AgentModelHttpFailure.status(error)).isEqualTo(status);
                        var classified = new AgentModelCallFailure(error);
                        assertThat(classified.retryable()).isEqualTo(retryable);
                        assertThat(classified.delay(Duration.ofSeconds(2))).isEqualTo(Duration.ofSeconds(30));
                    });
            assertThat(attempts).hasValue(1);
        } finally {
            server.stop(0);
        }
    }

    @ParameterizedTest
    @CsvSource({"server_error,true", "rate_limit_exceeded,true", "invalid_prompt,false", "insufficient_quota,false"})
    void failedResponsesEventsKeepTheProviderErrorWithoutInventingHttpStatus(String code, boolean retryable) throws Exception {
        String failure = event("response.failed", Map.of("sequence_number", 1, "response", Map.of("id", "failed-response",
                "status", "failed", "error", Map.of("code", code, "message", "request rejected"))));
        var server = endpoint(new AtomicReference<>(), failure);
        try {
            assertThatThrownBy(() -> model(server).stream(List.of(new UserMessage("test")), List.of(), null)
                    .collectList().block(Duration.ofSeconds(10)))
                    .satisfies(error -> {
                        assertThat(AgentModelHttpFailure.status(error)).isNull();
                        assertThat(AgentModelHttpFailure.hasFailedResponse(error)).isTrue();
                        assertThat(new AgentModelCallFailure(error).retryable()).isEqualTo(retryable);
                    });
        } finally {
            server.stop(0);
        }
    }

    private OpenAiResponsesAgentScopeModel model(HttpServer server) {
        String url = "http://127.0.0.1:" + server.getAddress().getPort();
        var api = ApiConfig.builder().apiUrl(url).autoAppendV1Path(true).build();
        return new OpenAiResponsesAgentScopeModel(api, "local-test-key", url, "test-model", null, true);
    }

    private HttpServer endpoint(AtomicReference<JsonNode> request, String response) throws Exception {
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            request.set(json.readTree(exchange.getRequestBody()));
            exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
            exchange.sendResponseHeaders(200, 0);
            try (var output = exchange.getResponseBody()) {
                output.write(response.getBytes(StandardCharsets.UTF_8));
            }
        });
        server.start();
        return server;
    }

    private String events(boolean invalid) throws Exception {
        return added("item-a", "call-a", 0, 1) + added("item-b", "call-b", 1, 2)
                + done("item-b", 1, 3, invalid ? "{broken" : "{\"projectId\":8}")
                + done("item-a", 0, 4, "{\"projectId\":4}")
                + event("response.completed", Map.of("sequence_number", 5, "response", Map.of(
                        "id", "resp-test", "status", "completed", "model", "test-model", "output", List.of(),
                        "usage", Map.of("input_tokens", 10, "output_tokens", 5, "total_tokens", 15))));
    }

    private String added(String id, String call, int index, int sequence) throws Exception {
        return event("response.output_item.added", Map.of("sequence_number", sequence, "output_index", index,
                "item", Map.of("id", id, "type", "function_call", "call_id", call, "name", "get_project",
                        "arguments", "", "status", "in_progress")));
    }

    private String done(String id, int index, int sequence, String arguments) throws Exception {
        return event("response.function_call_arguments.done", Map.of("sequence_number", sequence, "output_index", index,
                "item_id", id, "arguments", arguments, "name", "get_project"));
    }

    private String event(String type, Map<String, Object> data) throws Exception {
        var value = json.valueToTree(data);
        ((ObjectNode) value).put("type", type);
        return "event: " + type + "\ndata: " + json.writeValueAsString(value) + "\n\n";
    }
}
