package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.config.AgentScopeRuntimeProperties;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.state.AgentStateStore;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import reactor.test.StepVerifier;

import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentStatePreflightTests {

    private final AgentRuntimeSchedulers schedulers = schedulers();
    private final InMemoryStateStoreFailureGuard failures = new InMemoryStateStoreFailureGuard();

    @AfterEach
    void closeSchedulers() {
        schedulers.close();
    }

    @Test
    void clearsStaleMarkerAndChecksTheExactSlotOnStateScheduler() {
        AgentStateStore store = mock(AgentStateStore.class);
        RuntimeContext context = runtime("42", "afv:v2:conversation-7:assistant-v3");
        StateStoreSlot slot = new StateStoreSlot(context.getUserId(), context.getSessionId());
        failures.record(slot, "save", new IllegalStateException("stale"));
        AtomicReference<String> threadName = new AtomicReference<>();
        when(store.exists(context.getUserId(), context.getSessionId())).thenAnswer(ignored -> {
            threadName.set(Thread.currentThread().getName());
            return false;
        });
        AgentStatePreflight preflight = new AgentStatePreflight(store, failures, schedulers);

        StepVerifier.create(preflight.check(context)).verifyComplete();

        assertThat(threadName.get()).startsWith("agent-state-");
        assertThat(failures.failure(slot)).isEmpty();
        verify(store).exists(context.getUserId(), context.getSessionId());
    }

    @Test
    void deletesTheExactStaleStateSlotBeforeFailedRunRecovery() {
        AgentStateStore store = mock(AgentStateStore.class);
        AgentStatePreflight preflight = new AgentStatePreflight(store, failures, schedulers);

        StepVerifier.create(preflight.deleteSession(
                        "42", "afv:v2:conversation-7:assistant-v3"))
                .verifyComplete();

        verify(store).delete("42", "afv:v2:conversation-7:assistant-v3");
    }

    @Test
    void recordsAndPropagatesDelegateFailureWithoutFallingBack() {
        AgentStateStore delegate = mock(AgentStateStore.class);
        RuntimeContext context = runtime("42", "afv:v2:conversation-7:assistant-v3");
        StateStoreSlot slot = new StateStoreSlot(context.getUserId(), context.getSessionId());
        IllegalStateException redisFailure = new IllegalStateException("redis unavailable");
        when(delegate.exists(context.getUserId(), context.getSessionId())).thenThrow(redisFailure);
        AgentStateStore store = new FailClosedAgentStateStore(delegate, failures);
        AgentStatePreflight preflight = new AgentStatePreflight(store, failures, schedulers);

        StepVerifier.create(preflight.check(context))
                .expectErrorSatisfies(actual -> {
                    assertThat(actual).isInstanceOf(StateStoreFailure.class);
                    assertThat(actual).isSameAs(failures.failure(slot).orElseThrow());
                    assertThat(actual.getCause()).isSameAs(redisFailure);
                })
                .verify();
    }

    private RuntimeContext runtime(String userId, String sessionId) {
        return RuntimeContext.builder().userId(userId).sessionId(sessionId).build();
    }

    private AgentRuntimeSchedulers schedulers() {
        AgentScopeRuntimeProperties properties = new AgentScopeRuntimeProperties();
        properties.setStateThreads(1);
        properties.setJournalThreads(1);
        properties.setModelThreads(1);
        properties.setToolThreads(1);
        return new AgentRuntimeSchedulers(properties);
    }
}
