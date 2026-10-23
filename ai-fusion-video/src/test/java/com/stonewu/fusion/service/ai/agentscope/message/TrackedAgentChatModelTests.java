package com.stonewu.fusion.service.ai.agentscope.message;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.common.BusinessException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.agentscope.AgentScopeModelFactory;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelKey;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.provider.AiProviderContext;
import com.stonewu.fusion.service.ai.provider.AiProviderService;
import com.stonewu.fusion.service.ai.provider.AiProviderService.AgentScopeModelProvision;
import com.stonewu.fusion.service.ai.provider.OpenAiCompatibleAiProvider;
import com.stonewu.fusion.service.ai.provider.StreamingProviderFixtures;
import com.stonewu.fusion.service.ai.run.model.AgentModelRequests;
import com.stonewu.fusion.service.media.AgentMediaInputResolver;
import com.sun.net.httpserver.HttpServer;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.model.ChatModelBase;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Uses the real OpenAI formatter and single-send HTTP transport across reasoning rounds. */
class TrackedAgentChatModelTests {
    @ParameterizedTest
    @ValueSource(ints = {400, 401, 403, 429, 500, 503})
    void explicitHttpFailureFinishesTheOriginalRequestWithoutHoldingAnUnknownSubmission(int status) throws Exception {
        AtomicInteger sent = new AtomicInteger();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/chat/completions", exchange -> {
            exchange.getRequestBody().readAllBytes();
            sent.incrementAndGet();
            byte[] body = "{\"error\":{\"message\":\"Expected canvas tool is unavailable\"}}".getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(status, body.length);
            try (var output = exchange.getResponseBody()) { output.write(body); }
        });
        server.start();
        try {
            var model = AiModel.builder().id(7L).code("test-model").modelType(1).build();
            var api = ApiConfig.builder().id(1L).apiKey("local-test").autoAppendV1Path(true).build();
            var context = AiProviderContext.builder().apiConfig(api).apiKey("local-test").model(model)
                    .baseUrl("http://127.0.0.1:" + server.getAddress().getPort()).modelName("test-model")
                    .platform("openai_compatible").singleRequest(true).build();
            var ledger = mock(AgentModelRequests.class);
            var claim = new AgentModelRequests.Claim("request", "run", "worker", 1);
            when(ledger.admit(anyString(), eq(model), nullable(RuntimeContext.class), eq("fingerprint")))
                    .thenReturn(Optional.of(claim));
            try (var tracked = new TrackedAgentChatModel(new OpenAiCompatibleAiProvider().createAgentScopeModel(context),
                    () -> { throw new AssertionError("No automatic resend"); }, model, ledger, "fingerprint")) {
                assertThatThrownBy(() -> assertText(tracked, "创建分镜"))
                        .hasMessageContaining(String.valueOf(status)).hasMessageContaining("Expected canvas tool is unavailable");
            }
            assertThat(sent).hasValue(1);
            verify(ledger).failed(claim);
            verify(ledger, never()).completed(any(), any());
            verify(ledger, never()).uncertain(any(), eq(false));
            verify(ledger, never()).uncertain(any(), eq(true));
        } finally { server.stop(0); }
    }

    @ParameterizedTest
    @ValueSource(strings = {"length", "malformed", "array", "valid"})
    void receivedInvalidArgumentsNeverReachToolsButKeepTheirUsageAndDoNotResend(String scenario) throws Exception {
        AtomicInteger sent = new AtomicInteger();
        String arguments = switch (scenario) {
            case "length", "malformed" -> "{\"operations\":[{\"type\":\"canvas.node.create\"";
            case "array" -> "[]";
            default -> "{\"operations\":[]}";
        };
        byte[] body = StreamingProviderFixtures.sse(new ObjectMapper().writeValueAsString(Map.of("id", "output-validation", "object", "chat.completion.chunk",
                "choices", List.of(Map.of("index", 0, "finish_reason", scenario.equals("length") ? "length" : "tool_calls",
                        "delta", Map.of("role", "assistant", "tool_calls", List.of(Map.of(
                                "index", 0, "id", "call-1", "type", "function", "function", Map.of("name", "apply_canvas_operations", "arguments", arguments)))))),
                "usage", Map.of("prompt_tokens", 10, "completion_tokens", 2048, "total_tokens", 2058)))).getBytes(StandardCharsets.UTF_8);
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/chat/completions", exchange -> {
            exchange.getRequestBody().readAllBytes();
            sent.incrementAndGet();
            exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
            exchange.sendResponseHeaders(200, body.length);
            try (var output = exchange.getResponseBody()) { output.write(body); }
        });
        server.start();
        try {
            var model = AiModel.builder().id(7L).code("test-model").modelType(1).build();
            var api = ApiConfig.builder().id(1L).apiKey("local-test").autoAppendV1Path(true).build();
            var context = AiProviderContext.builder().apiConfig(api).apiKey("local-test").model(model)
                    .baseUrl("http://127.0.0.1:" + server.getAddress().getPort()).modelName("test-model")
                    .platform("openai_compatible").singleRequest(true).build();
            var ledger = mock(AgentModelRequests.class);
            var claim = new AgentModelRequests.Claim("request", "run", "worker", 1);
            when(ledger.admit(anyString(), eq(model), nullable(RuntimeContext.class), eq("fingerprint")))
                    .thenReturn(Optional.of(claim));
            try (var tracked = new TrackedAgentChatModel(new OpenAiCompatibleAiProvider().createAgentScopeModel(context),
                    () -> { throw new AssertionError("No automatic resend"); }, model, ledger, "fingerprint")) {
                var response = tracked.stream(List.of(Msg.builder().name("user").role(MsgRole.USER)
                        .content(TextBlock.builder().text("修改画布").build()).build()), List.of(), null).collectList();
                if (scenario.equals("valid")) assertThat(response.block(Duration.ofSeconds(8))).hasSize(1);
                else assertThatThrownBy(() -> response.block(Duration.ofSeconds(8)))
                        .isInstanceOf(BusinessException.class).hasMessageContaining(scenario.equals("length") ? "长度上限" : "参数不完整");
            }
            assertThat(sent).hasValue(1);
            verify(ledger).completed(eq(claim), any());
            verify(ledger, never()).uncertain(any(), eq(false));
            verify(ledger, never()).uncertain(any(), eq(true));
        } finally { server.stop(0); }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void eachRoundGetsANewTransportAndAnIndependentClaimWithoutHiddenResends(boolean secondFails) throws Exception {
        AtomicInteger sent = new AtomicInteger();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/chat/completions", exchange -> {
            exchange.getRequestBody().readAllBytes();
            int sequence = sent.incrementAndGet();
            int status = secondFails && sequence == 2 ? 503 : 200;
            String response = status == 200 ? """
                    {"id":"test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"已完成本轮"},
                    "finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}
                    """ : "{\"error\":{\"message\":\"test unavailable\"}}";
            byte[] bytes = (status == 200 ? StreamingProviderFixtures.sse(response.replace("\n", "")) : response).getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", status == 200 ? "text/event-stream" : "application/json");
            exchange.getResponseHeaders().set("Retry-After", "0");
            exchange.sendResponseHeaders(status, bytes.length);
            try (var output = exchange.getResponseBody()) { output.write(bytes); }
        });
        server.start();
        try {
            AiModel model = AiModel.builder().id(7L).code("ui-model").modelType(1).supportReasoning(false).build();
            String fingerprint = AgentKernelKey.modelRecordFingerprint(model);
            ApiConfig api = ApiConfig.builder().id(1L).apiKey("local-test").autoAppendV1Path(true).build();
            var context = AiProviderContext.builder().apiConfig(api).apiKey("local-test").model(model)
                    .baseUrl("http://127.0.0.1:" + server.getAddress().getPort()).modelName("ui-model")
                    .platform("openai_compatible").singleRequest(true).build();
            AiProviderService provider = mock(AiProviderService.class);
            when(provider.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call ->
                    new AgentScopeModelProvision(new OpenAiCompatibleAiProvider().createAgentScopeModel(context), fingerprint));
            AgentModelRequests ledger = mock(AgentModelRequests.class);
            List<String> claims = new CopyOnWriteArrayList<>();
            when(ledger.admit(anyString(), eq(model), nullable(RuntimeContext.class), eq(fingerprint))).thenAnswer(call -> {
                String id = call.getArgument(0);
                claims.add(id);
                return Optional.of(new AgentModelRequests.Claim(id, "test-run", "worker", 1));
            });
            var key = AgentKernelKey.create("writer", fingerprint, AgentKernelKey.promptVersion("Write"), List.of(), "test");
            var spec = new AgentKernelSpec(key, model, "writer", "Writer", "Writes", "Write", 10, List.of(), Set.of(), "test");
            var factory = new AgentScopeModelFactory(provider, mock(AgentMediaInputResolver.class), ledger);
            try (var owned = factory.create(spec)) {
                assertText(owned.model(), "第一轮推理");
                if (secondFails) {
                    assertThatThrownBy(() -> assertText(owned.model(), "第二轮推理"))
                            .isInstanceOf(RuntimeException.class);
                    assertThat(sent).hasValue(2);
                    verify(ledger).completed(any(), any());
                    verify(ledger).failed(any());
                    verify(ledger, never()).uncertain(any(), eq(false));
                } else {
                    assertText(owned.model(), "上下文摘要请求");
                    assertText(owned.model(), "摘要后的下一轮推理");
                    assertThat(sent).hasValue(3);
                    verify(ledger, times(3)).completed(any(), any());
                    verify(ledger, never()).uncertain(any(), any(Boolean.class));
                }
                assertThat(claims).hasSize(sent.get()).doesNotHaveDuplicates();
                verify(provider, times(sent.get())).provisionSingleRequestModel(model);
            }
        } finally { server.stop(0); }
    }

    private static void assertText(ChatModelBase model, String prompt) {
        var response = model.stream(List.of(Msg.builder().name("user").role(MsgRole.USER)
                        .content(TextBlock.builder().text(prompt).build()).build()), List.of(), null)
                .collectList().block(Duration.ofSeconds(8));
        assertThat(response).isNotEmpty();
        assertThat(response.stream().flatMap(item -> item.getContent().stream()).filter(TextBlock.class::isInstance)
                .map(TextBlock.class::cast).map(TextBlock::getText).reduce("", String::concat)).isEqualTo("已完成本轮");
    }
}
