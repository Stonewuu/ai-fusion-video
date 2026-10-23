package com.stonewu.fusion.service.ai.agentscope.message;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.run.model.AgentModelRequests;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.ChatUsage;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import io.agentscope.core.model.transport.HttpTransportException;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;
import reactor.test.StepVerifier;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.timeout;

class TrackedAgentStreamingTests {
    private final AiModel model = AiModel.builder().id(7L).build();
    private final AgentModelRequests ledger = mock(AgentModelRequests.class);
    private final AgentModelRequests.Claim claim = new AgentModelRequests.Claim("request", "run", "owner", 1);
    private final ChatUsage usage = ChatUsage.builder().inputTokens(10).outputTokens(5).build();

    @Test
    void forwardsTextAndIncompleteToolChunksBeforeTheResponseCompletes() {
        Sinks.Empty<Void> finish = Sinks.empty();
        ChatResponse text = text("第一段");
        ChatResponse arguments = tool("first", "write", "{\"value\":");
        try (var tracked = tracked(Flux.concat(Flux.just(text, arguments), finish.asMono()
                .thenMany(Flux.just(tool("first", "write", "1}"), terminal("tool_calls")))))) {
            StepVerifier.create(tracked.stream(List.of(), List.of(), null))
                    .expectNext(text, arguments)
                    .then(() -> {
                        verify(ledger, never()).completed(any(), any());
                        finish.tryEmitEmpty();
                    })
                    .expectNextCount(2).expectComplete().verify(Duration.ofSeconds(3));
            verify(ledger).completed(claim, usage);
        }
    }

    @Test
    void keepsParallelToolArgumentsSeparateAndRejectsTruncatedOutputAfterVisibleText() {
        try (var tracked = tracked(Flux.just(text("正在处理"), tool("a", "one", "{\"value\":"),
                tool("b", "two", "{\"value\":"), tool("a", "one", "1}"), tool("b", "two", "2"), terminal("tool_calls")))) {
            StepVerifier.create(tracked.stream(List.of(), List.of(), null))
                    .expectNextCount(6)
                    .expectErrorSatisfies(error -> assertThat(error).isInstanceOf(BusinessException.class).hasMessageContaining("two"))
                    .verify(Duration.ofSeconds(3));
            verify(ledger).completed(claim, usage);
            verify(ledger, never()).uncertain(any(), any(Boolean.class));
        }
    }

    @Test
    void eofWithoutACompletionMarkerIsUncertainAndDoesNotCompleteTheAgentTurn() {
        try (var tracked = tracked(Flux.just(text("半段正文")))) {
            StepVerifier.create(tracked.stream(List.of(), List.of(), null)).expectNextCount(1)
                    .expectErrorMessage("模型连接在返回完成标记前中断，未执行本轮工具")
                    .verify(Duration.ofSeconds(3));
            verify(ledger).uncertain(claim, false);
            verify(ledger, never()).completed(any(), any());
        }
    }

    @Test
    void aDisconnectIsStillUncertainEvenWhenItsMessageMentionsAnHttpStatus() {
        var disconnected = new HttpTransportException("Connection reset after HTTP status 500 was mentioned in content");
        try (var tracked = tracked(Flux.concat(Flux.just(text("半段内容")), Flux.error(disconnected)))) {
            StepVerifier.create(tracked.stream(List.of(), List.of(), null)).expectNextCount(1)
                    .expectErrorMatches(error -> error == disconnected).verify(Duration.ofSeconds(3));
            verify(ledger).uncertain(claim, false);
            verify(ledger, never()).failed(any());
        }
    }

    @Test
    void cancellingAfterTheFirstChunkCancelsTheProviderAndRecordsTheOriginalRequest() {
        AtomicBoolean cancelled = new AtomicBoolean();
        try (var tracked = tracked(Flux.concat(Flux.just(text("已到达")), Flux.<ChatResponse>never())
                .doOnCancel(() -> cancelled.set(true)))) {
            StepVerifier.create(tracked.stream(List.of(), List.of(), null))
                    .expectNextCount(1).thenCancel().verify(Duration.ofSeconds(3));
            assertThat(cancelled).isTrue();
            verify(ledger, timeout(2000)).uncertain(claim, true);
        }
    }

    private TrackedAgentChatModel tracked(Flux<ChatResponse> stream) {
        when(ledger.admit(anyString(), any(), any(), anyString())).thenReturn(Optional.of(claim));
        ChatModelBase delegate = new ChatModelBase() {
            @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
                return stream;
            }
            @Override public String getModelName() { return "test"; }
        };
        return new TrackedAgentChatModel(delegate, () -> { throw new AssertionError("不能重发原请求"); }, model, ledger, "test");
    }

    private ChatResponse text(String text) {
        return ChatResponse.builder().content(List.of(TextBlock.builder().text(text).build())).build();
    }

    private ChatResponse tool(String id, String name, String arguments) {
        return ChatResponse.builder().content(List.of(ToolUseBlock.builder().id(id).name(name)
                .input(Map.of()).content(arguments).build())).build();
    }

    private ChatResponse terminal(String reason) {
        return ChatResponse.builder().content(List.of()).finishReason(reason).usage(usage).build();
    }
}
