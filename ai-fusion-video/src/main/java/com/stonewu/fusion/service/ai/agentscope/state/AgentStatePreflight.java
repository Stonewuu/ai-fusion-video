package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.state.AgentStateStore;
import reactor.core.publisher.Mono;

import java.util.Objects;

public final class AgentStatePreflight {

    private final AgentStateStore store;
    private final StateStoreFailureGuard failures;
    private final AgentRuntimeSchedulers schedulers;
    public AgentStatePreflight(
            AgentStateStore store,
            StateStoreFailureGuard failures,
            AgentRuntimeSchedulers schedulers) {
        this.store = Objects.requireNonNull(store, "store must not be null");
        this.failures = Objects.requireNonNull(failures, "failures must not be null");
        this.schedulers = Objects.requireNonNull(schedulers, "schedulers must not be null");
    }

    public Mono<Void> check(RuntimeContext context) {
        return Mono.fromRunnable(() -> {
                    RuntimeContext safeContext = Objects.requireNonNull(
                            context, "context must not be null");
                    StateStoreSlot slot = new StateStoreSlot(
                            safeContext.getUserId(), safeContext.getSessionId());
                    failures.clear(slot);
                    store.exists(slot.userId(), slot.sessionId());
                    failures.throwIfFailed(slot);
                })
                .subscribeOn(schedulers.state())
                .then();
    }

    public Mono<Void> deleteSession(String runtimeUserId, String sessionId) {
        return Mono.fromRunnable(() -> deleteWholeSession(
                        requireText(runtimeUserId, "runtimeUserId"),
                        requireText(sessionId, "sessionId")))
                .subscribeOn(schedulers.state())
                .then();
    }

    private void deleteWholeSession(String userId, String sessionId) {
        StateStoreSlot slot = new StateStoreSlot(userId, sessionId);
        failures.clear(slot);
        store.delete(slot.userId(), slot.sessionId());
        failures.throwIfFailed(slot);
    }

    private String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value.trim();
    }
}
