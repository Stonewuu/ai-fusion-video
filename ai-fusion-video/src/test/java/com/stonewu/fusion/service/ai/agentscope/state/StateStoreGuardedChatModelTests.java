package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.AgentScopeRuntimeProperties;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import io.agentscope.core.agent.AgentBase;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import io.agentscope.core.state.AgentStateStore;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import reactor.test.StepVerifier;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class StateStoreGuardedChatModelTests {

    @Test
    void delegatesOnlyWhenRuntimeSlotHasNoFailureAndPreservesCapabilities() {
        InMemoryStateStoreFailureGuard failures = new InMemoryStateStoreFailureGuard();
        CountingChatModel delegate = new CountingChatModel();
        StateStoreGuardedChatModel guarded = new StateStoreGuardedChatModel(delegate, failures);
        RuntimeContext runtime = runtime();

        StepVerifier.create(guarded.stream(List.of(), List.of(), GenerateOptions.builder().build())
                .contextWrite(context -> context.put(AgentBase.RUNTIME_CONTEXT_KEY, runtime)))
                .assertNext(response -> assertThat(((TextBlock) response.getContent().getFirst()).getText())
                        .isEqualTo("ok"))
                .verifyComplete();

        assertThat(delegate.invocations()).isEqualTo(1);
        assertThat(guarded.getModelName()).isEqualTo(delegate.getModelName());
        assertThat(guarded.getContextWindowSize()).isEqualTo(8_192);
        assertThat(guarded.supportsNativeStructuredOutput()).isTrue();
        assertThat(guarded.supportsNativeStructuredOutputWithTools()).isTrue();
    }

    @Test
    void exposesTheConfiguredModelContextWindowToHarnessMiddleware() {
        CountingChatModel delegate = new CountingChatModel();
        StateStoreGuardedChatModel guarded = new StateStoreGuardedChatModel(
                delegate, new InMemoryStateStoreFailureGuard(), 262_144);

        assertThat(delegate.getContextWindowSize()).isEqualTo(8_192);
        assertThat(guarded.getContextWindowSize()).isEqualTo(262_144);
    }

    @Test
    void rejectsStoredFailureBeforeInvokingProvider() {
        InMemoryStateStoreFailureGuard failures = new InMemoryStateStoreFailureGuard();
        RuntimeContext runtime = runtime();
        StateStoreSlot slot = new StateStoreSlot(runtime.getUserId(), runtime.getSessionId());
        StateStoreFailure stored = failures.record(slot, "get", new IllegalStateException("redis down"));
        CountingChatModel delegate = new CountingChatModel();
        StateStoreGuardedChatModel guarded = new StateStoreGuardedChatModel(delegate, failures);

        StepVerifier.create(guarded.stream(List.of(), List.of(), GenerateOptions.builder().build())
                        .contextWrite(context -> context.put(AgentBase.RUNTIME_CONTEXT_KEY, runtime)))
                .expectErrorSatisfies(actual -> assertThat(actual).isSameAs(stored))
                .verify();

        assertThat(delegate.invocations()).isZero();
    }

    @Test
    void rejectsMissingRuntimeContextBeforeInvokingProvider() {
        CountingChatModel delegate = new CountingChatModel();
        StateStoreGuardedChatModel guarded = new StateStoreGuardedChatModel(
                delegate, new InMemoryStateStoreFailureGuard());

        StepVerifier.create(guarded.stream(List.of(), List.of(), GenerateOptions.builder().build()))
                .expectErrorSatisfies(actual -> {
                    assertThat(actual).isInstanceOf(BusinessException.class);
                    assertThat(((BusinessException) actual).getCode()).isEqualTo(500);
                    assertThat(actual.getMessage()).isEqualTo("STATE_STORE_FAILED: RuntimeContext missing");
                })
                .verify();

        assertThat(delegate.invocations()).isZero();
    }

    @Test
    void swallowedStateLoadCannotReachProviderAfterPreflightFailure() {
        AgentStateStore delegateStore = mock(AgentStateStore.class);
        InMemoryStateStoreFailureGuard failures = new InMemoryStateStoreFailureGuard();
        RuntimeContext runtime = runtime();
        when(delegateStore.exists(runtime.getUserId(), runtime.getSessionId()))
                .thenThrow(new IllegalStateException("redis down"));
        CountingChatModel delegateModel = new CountingChatModel();
        StateStoreGuardedChatModel guarded = new StateStoreGuardedChatModel(delegateModel, failures);
        AgentScopeRuntimeProperties properties = new AgentScopeRuntimeProperties();
        properties.setStateThreads(1);
        properties.setJournalThreads(1);
        properties.setModelThreads(1);
        properties.setToolThreads(1);

        try (AgentRuntimeSchedulers schedulers = new AgentRuntimeSchedulers(properties)) {
            AgentStatePreflight preflight = new AgentStatePreflight(
                    new FailClosedAgentStateStore(delegateStore, failures), failures, schedulers);

            StepVerifier.create(preflight.check(runtime)
                            .thenMany(guarded.stream(List.of(), List.of(), GenerateOptions.builder().build())
                                    .contextWrite(context -> context.put(AgentBase.RUNTIME_CONTEXT_KEY, runtime))))
                    .expectError(StateStoreFailure.class)
                    .verify();
        }

        assertThat(delegateModel.invocations()).isZero();
    }

    private RuntimeContext runtime() {
        return RuntimeContext.builder()
                .userId("42")
                .sessionId("afv:v2:conversation-7:assistant-v3")
                .build();
    }

    private static final class CountingChatModel extends ChatModelBase {

        private final AtomicInteger invocations = new AtomicInteger();

        private CountingChatModel() {
            setContextWindowSize(8_192);
            setNativeStructuredOutput(true);
            setNativeStructuredOutputWithTools(true);
        }

        @Override
        public String getModelName() {
            return "counting";
        }

        @Override
        protected Flux<ChatResponse> doStream(
                List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            return Flux.defer(() -> {
                invocations.incrementAndGet();
                return Flux.just(ChatResponse.builder()
                        .content(List.of(TextBlock.builder().text("ok").build()))
                        .finishReason("stop")
                        .build());
            });
        }

        private int invocations() {
            return invocations.get();
        }
    }
}
