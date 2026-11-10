package com.stonewu.fusion.service.ai.agentscope.message;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.agentscope.AgentScopeToolAdapter;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.CancellationContext;
import com.stonewu.fusion.service.ai.agentscope.context.ToolExecutionContext;
import com.stonewu.fusion.service.ai.agentscope.permission.AgentToolPermissionPolicy;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.agentscope.state.InMemoryStateStoreFailureGuard;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.run.RunLeaseGuard;
import com.stonewu.fusion.service.ai.run.model.AgentModelRequests;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.AgentBase;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.event.AgentResultEvent;
import io.agentscope.core.event.CustomEvent;
import io.agentscope.core.event.ModelCallStartEvent;
import io.agentscope.core.event.ModelCallEndEvent;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.UserMessage;
import io.agentscope.core.middleware.AgentInput;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import io.agentscope.core.state.AgentState;
import io.agentscope.core.state.InMemoryAgentStateStore;
import io.agentscope.core.tool.Toolkit;
import io.agentscope.harness.agent.HarnessAgent;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import reactor.test.StepVerifier;

import java.net.SocketTimeoutException;
import java.time.Duration;
import java.time.Instant;
import java.time.ZonedDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** 运行 SDK 的完整工具循环，验证重试保留原会话、流式输出及已保存的工具结果。 */
class AgentModelRetryMiddlewareTests {
    final InMemoryStateStoreFailureGuard failures = new InMemoryStateStoreFailureGuard();
    final AgentModelRetryMiddleware retries = new AgentModelRetryMiddleware(failures,
            List.of(Duration.ZERO, Duration.ZERO, Duration.ZERO));
    final RuntimeContext runtime = RuntimeContext.builder().userId("7").sessionId("same-child-session")
            .put(AgentRunContext.class, new AgentRunContext("same-child-run", "worker", 1, Instant.now().plusSeconds(60)))
            .put(CancellationContext.class, CancellationContext.noop())
            .put(ToolExecutionContext.class, new ToolExecutionContext(7L, 1, 7L, null)).build();

    @Test
    void midStreamFailureResumesTheSameChildWithoutRepeatingTheSuccessfulWrite() {
        var requests = mock(AgentModelRequests.class);
        var model = AiModel.builder().id(32L).code("retry-test").build();
        List<String> admissions = new CopyOnWriteArrayList<>();
        when(requests.admit(anyString(), eq(model), any(), eq("fixed"))).thenAnswer(call -> {
            RuntimeContext context = call.getArgument(2);
            assertThat(context.getSessionId()).isEqualTo("same-child-session");
            assertThat(context.get(AgentRunContext.class).runId()).isEqualTo("same-child-run");
            String id = call.getArgument(0);
            admissions.add(id);
            return Optional.of(new AgentModelRequests.Claim(id, "same-child-run", "worker", 1));
        });
        var rounds = new AtomicInteger();
        List<List<Msg>> inputs = new CopyOnWriteArrayList<>();
        var state = new InMemoryAgentStateStore();
        ToolExecutor write = mock(ToolExecutor.class);
        when(write.getToolName()).thenReturn("save_test");
        when(write.getToolDescription()).thenReturn("保存测试内容");
        when(write.getParametersSchema()).thenReturn("{\"type\":\"object\",\"properties\":{\"value\":{\"type\":\"integer\"}},\"required\":[\"value\"]}");
        when(write.execute(anyString(), any())).thenReturn("{\"saved\":true}");
        var lease = mock(RunLeaseGuard.class);
        when(lease.assertLease(anyString(), anyString(), anyLong())).thenReturn(Mono.empty());
        var toolkit = new Toolkit();
        toolkit.registerAgentTool(new AgentScopeToolAdapter(write, AgentScopeToolSchema.prepare(new ObjectMapper(),
                write.getParametersSchema(), "save_test"), Schedulers.boundedElastic(), lease, new ObjectMapper(), model, List.of()));
        try (var tracked = new TrackedAgentChatModel(new ScriptedModel(rounds, inputs),
                () -> new ScriptedModel(rounds, inputs), model, requests, "fixed")) {
            var agent = HarnessAgent.builder().name("子智能体").sysPrompt("使用工具保存一次，然后总结")
                    .model(tracked).stateStore(state).toolkit(toolkit).middleware(retries).maxIters(8).maxRetries(1)
                    .disableFilesystemTools().disableToolResultEviction().disableShellTool().disableWebTools()
                    .disableMemoryTools().disableMemoryHooks().disableTranscript().enableAgentTracingLog(false)
                    .disableSessionPersistence().disableWorkspaceContext().disableAtPathExpansion()
                    .disableSubagents().disableDynamicSubagents().disableToolsConfig().disableDefaultWorkspaceSkills()
                    .permissionContext(AgentToolPermissionPolicy.contextFor(toolkit, ToolExecutionMode.FULL_ACCESS)).build();
            var events = agent.streamEvents(List.of(new UserMessage("保存并总结")), runtime)
                    .collectList().block(Duration.ofSeconds(10));
            assertThat(events).filteredOn(CustomEvent.class::isInstance).hasSize(1);
            assertThat(events).filteredOn(AgentResultEvent.class::isInstance).singleElement()
                    .satisfies(event -> assertThat(((AgentResultEvent) event).getResult().getTextContent()).isEqualTo("已续跑完成"));
            assertThat(inputs).hasSize(3);
            List<Msg> resumed = inputs.get(2);
            assertThat(resumed.stream().filter(msg -> msg.getRole() == MsgRole.USER)).hasSize(1);
            assertThat(resumed.stream().flatMap(msg -> msg.getContent().stream()).filter(ToolResultBlock.class::isInstance))
                    .hasSize(1);
            assertThat(resumed).allSatisfy(message -> assertThat(message.getTextContent()).doesNotContain("这段未完成"));
            assertThat(resumed.stream().flatMap(message -> message.getContent().stream())
                    .filter(ToolUseBlock.class::isInstance).map(ToolUseBlock.class::cast).map(ToolUseBlock::getId))
                    .containsExactly("saved-call");
            assertThat(state.get("7", "same-child-session", "agent_state", AgentState.class)).isPresent();
            verify(write, times(1)).execute(anyString(), any());
            verify(requests, times(2)).completed(any(), any());
            verify(requests).uncertain(any(), eq(false));
            assertThat(admissions).hasSize(3).doesNotHaveDuplicates();
        }
    }

    @Test
    void retriesHaveABoundAndDoNotResubmitTheOriginalUserInput() {
        var calls = new AtomicInteger();
        List<AgentInput> inputs = new CopyOnWriteArrayList<>();
        assertThatThrownBy(() -> retries.onAgent(mock(Agent.class), runtime, new AgentInput(List.of(new UserMessage("任务"))), input -> {
            inputs.add(input);
            calls.incrementAndGet();
            return Flux.concat(Flux.just(new ModelCallStartEvent("failed-reply")), Flux.error(
                    new AgentModelCallFailure(GenerationProviderFailure.http(520, "upstream unavailable"))));
        }).collectList().block(Duration.ofSeconds(3))).isInstanceOf(AgentModelRetryMiddleware.AgentModelRetryExhausted.class);
        assertThat(calls).hasValue(4);
        assertThat(inputs.getFirst().msgs()).hasSize(1);
        assertThat(inputs.subList(1, inputs.size())).allSatisfy(input -> assertThat(input.msgs()).isEmpty());
    }

    @ParameterizedTest
    @ValueSource(ints = {400, 401, 402, 403, 404, 422})
    void permanentProviderErrorsAreNotRetried(int status) {
        var calls = new AtomicInteger();
        StepVerifier.create(retries.onAgent(mock(Agent.class), runtime, new AgentInput(List.of()), input -> {
            calls.incrementAndGet();
            return Flux.error(new AgentModelCallFailure(GenerationProviderFailure.http(status, "configuration error")));
        })).expectError(AgentModelCallFailure.class).verify();
        assertThat(calls).hasValue(1);
    }

    @Test
    void exhaustedQuotaIsNotRetriedButPerMinuteRateLimitsAre() {
        assertThat(new AgentModelCallFailure(GenerationProviderFailure.http(429, "insufficient_quota: check billing")).retryable()).isFalse();
        assertThat(new AgentModelCallFailure(GenerationProviderFailure.http(429, "quota exceeded: requests per minute")).retryable()).isTrue();
    }

    @Test
    void cancellationDuringBackoffStopsWithoutAnotherModelCall() {
        var calls = new AtomicInteger();
        var middleware = new AgentModelRetryMiddleware(failures, List.of(Duration.ofSeconds(10)));
        StepVerifier.create(middleware.onAgent(mock(Agent.class), runtime, new AgentInput(List.of()), input -> {
            calls.incrementAndGet();
            return Flux.error(new AgentModelCallFailure(new SocketTimeoutException("timeout")));
        })).expectNextMatches(CustomEvent.class::isInstance).thenCancel().verify(Duration.ofSeconds(2));
        assertThat(calls).hasValue(1);
    }

    @Test
    void failedStateSaveStopsRetryRatherThanResumingStaleContext() {
        var failure = failures.record(new StateStoreSlot("7", "same-child-session"), "save", new IllegalStateException("database unavailable"));
        StepVerifier.create(retries.onAgent(mock(Agent.class), runtime, new AgentInput(List.of()), input ->
                Flux.error(new AgentModelCallFailure(new SocketTimeoutException("timeout")))))
                .expectErrorMatches(error -> error == failure).verify();
    }

    @Test
    void providerBackoffIsRespectedAndLongWaitsDoNotTriggerEarlyRequests() {
        String date = ZonedDateTime.now(ZoneOffset.UTC).plusSeconds(30).format(DateTimeFormatter.RFC_1123_DATE_TIME);
        var dated = new AgentModelCallFailure(GenerationProviderFailure.http(429, "rate limited", date));
        assertThat(dated.delay(Duration.ofSeconds(2))).isBetween(Duration.ofSeconds(25), Duration.ofSeconds(30));
        var malformed = new AgentModelCallFailure(GenerationProviderFailure.http(429, "rate limited", "invalid"));
        assertThat(malformed.delay(Duration.ofSeconds(2))).isEqualTo(Duration.ofSeconds(2));
        var deferred = new AgentModelCallFailure(GenerationProviderFailure.http(429, "rate limited", "300"));
        assertThat(deferred.retryable()).isFalse();
    }

    @Test
    void manyIntermittentFailuresDoNotAccumulateNestedRetrySubscriptions() {
        var calls = new AtomicInteger();
        Long notices = retries.onAgent(mock(Agent.class), runtime, new AgentInput(List.of(new UserMessage("长任务"))), input -> {
            int round = calls.incrementAndGet();
            if (round > 1) {
                assertThat(input.msgs()).isEmpty();
            }
            if (round > 500) {
                return Flux.just(new AgentResultEvent(Msg.builder().role(MsgRole.ASSISTANT).textContent("完成").build()));
            }
            return Flux.<AgentEvent>just(new ModelCallEndEvent("success-" + round, null), new ModelCallStartEvent("failed-" + round))
                    .concatWith(Flux.error(new AgentModelCallFailure(new SocketTimeoutException("timeout"))));
        }).limitRate(1).filter(CustomEvent.class::isInstance).count().block(Duration.ofSeconds(10));
        assertThat(notices).isEqualTo(500L);
        assertThat(calls).hasValue(501);
    }

    private static final class ScriptedModel extends ChatModelBase {
        final AtomicInteger rounds;
        final List<List<Msg>> inputs;

        ScriptedModel(AtomicInteger rounds, List<List<Msg>> inputs) {
            this.rounds = rounds;
            this.inputs = inputs;
        }

        @Override
        protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            inputs.add(List.copyOf(messages));
            int round = rounds.incrementAndGet();
            if (round == 1) {
                return Flux.just(ChatResponse.builder().content(List.of(ToolUseBlock.builder().id("saved-call").name("save_test")
                        .input(Map.of("value", 1)).content("{\"value\":1}").build())).finishReason("tool_calls").build());
            }
            if (round == 2) {
                return Flux.concat(Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder().text("这段未完成").build(),
                        ToolUseBlock.builder().id("discarded-call").name("save_test").content("{\"value\":").build())).build()),
                        Flux.error(new SocketTimeoutException("read timeout")));
            }
            return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder().text("已续跑完成").build())).finishReason("stop").build());
        }

        @Override
        public String getModelName() {
            return "retry-test";
        }
    }
}
