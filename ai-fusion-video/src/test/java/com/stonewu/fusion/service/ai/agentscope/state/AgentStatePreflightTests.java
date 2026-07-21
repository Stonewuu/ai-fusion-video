package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.config.AgentScopeRuntimeProperties;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.state.AgentStateStore;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import reactor.test.StepVerifier;

import java.util.List;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
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

    @Test
    void deletesOnlyMatchingConversationSessionsSequentiallyOnStateScheduler() {
        AgentStateStore store = mock(AgentStateStore.class);
        when(store.listSessionIds("42")).thenReturn(Set.of(
                "afv:v2:conversation-7:assistant-v3",
                "afv:v2:conversation-7:asset-agent-v1",
                "afv:v2:conversation-8:assistant-v3"));
        List<String> deleted = new CopyOnWriteArrayList<>();
        List<String> threads = new CopyOnWriteArrayList<>();
        doAnswer(invocation -> {
            deleted.add(invocation.getArgument(1));
            threads.add(Thread.currentThread().getName());
            return null;
        }).when(store).delete(org.mockito.ArgumentMatchers.eq("42"), org.mockito.ArgumentMatchers.anyString());
        AgentStatePreflight preflight = new AgentStatePreflight(store, failures, schedulers);

        StepVerifier.create(preflight.deleteConversationSessions("42", "conversation-7"))
                .verifyComplete();

        assertThat(deleted).containsExactlyInAnyOrder(
                "afv:v2:conversation-7:assistant-v3",
                "afv:v2:conversation-7:asset-agent-v1");
        assertThat(threads).allSatisfy(name -> assertThat(name).startsWith("agent-state-"));
        verify(store, never()).delete("42", "afv:v2:conversation-8:assistant-v3");
    }

    @Test
    void sessionCleanupFailsClosedAndStopsAtFirstDeleteFailure() {
        AgentStateStore delegate = mock(AgentStateStore.class);
        when(delegate.listSessionIds("42")).thenReturn(Set.of("afv:v2:conversation-7:assistant-v3"));
        IllegalStateException redisFailure = new IllegalStateException("delete failed");
        doThrow(redisFailure).when(delegate).delete("42", "afv:v2:conversation-7:assistant-v3");
        AgentStateStore store = new FailClosedAgentStateStore(delegate, failures);
        AgentStatePreflight preflight = new AgentStatePreflight(store, failures, schedulers);

        StepVerifier.create(preflight.deleteConversationSessions("42", "conversation-7"))
                .expectErrorSatisfies(actual -> {
                    assertThat(actual).isInstanceOf(StateStoreFailure.class);
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
