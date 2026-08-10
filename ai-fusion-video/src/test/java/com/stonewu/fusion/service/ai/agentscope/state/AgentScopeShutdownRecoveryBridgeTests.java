package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.config.AgentScopeRuntimeProperties;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEndEvent;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.UserMessage;
import io.agentscope.core.middleware.AgentInput;
import io.agentscope.core.state.AgentState;
import io.agentscope.core.state.AgentStateStore;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import reactor.test.StepVerifier;

import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

class AgentScopeShutdownRecoveryBridgeTests {

    private final AgentRuntimeSchedulers schedulers = schedulers();
    private final AgentStateStore store = mock(AgentStateStore.class);
    private final AgentScopeShutdownRecoveryBridge bridge =
            new AgentScopeShutdownRecoveryBridge(store, schedulers);

    @AfterEach
    void closeSchedulers() {
        schedulers.close();
    }

    @Test
    void acknowledgesTheNativeShutdownRetryMarkerOnSuccess() {
        Fixture fixture = fixture();

        StepVerifier.create(invoke(fixture, Flux.just(new AgentEndEvent("reply"))))
                .expectNextCount(1)
                .verifyComplete();

        assertThat(fixture.state().isShutdownInterrupted()).isFalse();
        verify(store).save(
                fixture.runtime().getUserId(),
                fixture.runtime().getSessionId(),
                "agent_state",
                fixture.state());
    }

    @Test
    void doesNotAcknowledgeWhenTheAgentCompletesWithoutASuccessTerminalEvent() {
        Fixture fixture = fixture();

        StepVerifier.create(invoke(fixture, Flux.empty()))
                .verifyComplete();

        assertThat(fixture.state().isShutdownInterrupted()).isTrue();
        verify(store, never()).save(
                fixture.runtime().getUserId(),
                fixture.runtime().getSessionId(),
                "agent_state",
                fixture.state());
    }

    @Test
    void keepsThePersistedMarkerUntouchedWhenTheAgentFails() {
        Fixture fixture = fixture();

        StepVerifier.create(invoke(
                        fixture, Flux.error(new IllegalStateException("model failed"))))
                .expectErrorMessage("model failed")
                .verify();

        assertThat(fixture.state().isShutdownInterrupted()).isTrue();
        verify(store, never()).save(
                fixture.runtime().getUserId(),
                fixture.runtime().getSessionId(),
                "agent_state",
                fixture.state());
    }

    @Test
    void keepsThePersistedMarkerUntouchedOnCancellation() {
        Fixture fixture = fixture();

        StepVerifier.create(invoke(fixture, Flux.never()))
                .thenCancel()
                .verify();

        verify(store, never()).save(
                fixture.runtime().getUserId(),
                fixture.runtime().getSessionId(),
                "agent_state",
                fixture.state());
        assertThat(fixture.state().isShutdownInterrupted()).isTrue();
    }

    @Test
    void failsTheSuccessfulCallAndRestoresTheMarkerWhenAcknowledgementCannotPersist() {
        Fixture fixture = fixture();
        doThrow(new IllegalStateException("redis unavailable"))
                .when(store)
                .save(
                        fixture.runtime().getUserId(),
                        fixture.runtime().getSessionId(),
                        "agent_state",
                        fixture.state());

        StepVerifier.create(invoke(fixture, Flux.just(new AgentEndEvent("reply"))))
                .expectErrorSatisfies(error -> assertThat(error)
                        .hasMessage("redis unavailable"))
                .verify();

        assertThat(fixture.state().isShutdownInterrupted()).isTrue();
    }

    private Flux<io.agentscope.core.event.AgentEvent> invoke(
            Fixture fixture,
            Flux<io.agentscope.core.event.AgentEvent> result) {
        return bridge.onAgent(
            mock(Agent.class),
            fixture.runtime(),
            new AgentInput(List.of(fixture.persisted(), fixture.retried())),
            prepared -> bridge.onSystemPrompt(
                            mock(Agent.class), fixture.runtime(), "system")
                    .doOnNext(ignored -> fixture.state().setShutdownInterrupted(false))
                    .thenMany(result));
    }

    private Fixture fixture() {
        Msg persisted = new UserMessage("persisted request");
        Msg retried = new UserMessage("retried request");
        AgentState state = AgentState.builder()
                .userId("42")
                .sessionId("afv:v2:conversation-7:assistant-v3")
                .context(List.of(persisted))
                .shutdownInterrupted(true)
                .build();
        RuntimeContext runtime = RuntimeContext.builder()
                .userId(state.getUserId())
                .sessionId(state.getSessionId())
                .agentState(state)
                .build();
        org.mockito.Mockito.when(store.get(
                        state.getUserId(),
                        state.getSessionId(),
                        "agent_state",
                        AgentState.class))
                .thenReturn(Optional.of(state));
        return new Fixture(
                state,
                runtime,
                persisted,
                retried);
    }

    private AgentRuntimeSchedulers schedulers() {
        AgentScopeRuntimeProperties properties = new AgentScopeRuntimeProperties();
        properties.setStateThreads(1);
        properties.setJournalThreads(1);
        properties.setModelThreads(1);
        properties.setToolThreads(1);
        return new AgentRuntimeSchedulers(properties);
    }

    private record Fixture(
            AgentState state,
            RuntimeContext runtime,
            Msg persisted,
            Msg retried) {
    }
}
