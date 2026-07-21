package com.stonewu.fusion.service.ai.agentscope.state;

import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.state.State;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Supplier;

public final class FailClosedAgentStateStore implements AgentStateStore {

    private final AgentStateStore delegate;
    private final StateStoreFailureGuard failures;

    public FailClosedAgentStateStore(AgentStateStore delegate, StateStoreFailureGuard failures) {
        this.delegate = Objects.requireNonNull(delegate, "delegate must not be null");
        this.failures = Objects.requireNonNull(failures, "failures must not be null");
    }

    @Override
    public void save(String userId, String sessionId, String key, State state) {
        guarded(userId, sessionId, "save", () -> delegate.save(userId, sessionId, key, state));
    }

    @Override
    public void save(
            String userId, String sessionId, String key, List<? extends State> states) {
        guarded(userId, sessionId, "save", () -> delegate.save(userId, sessionId, key, states));
    }

    @Override
    public <T extends State> Optional<T> get(
            String userId, String sessionId, String key, Class<T> stateType) {
        if (isFrameworkAnonymousGet(userId, sessionId, key)) {
            return Optional.empty();
        }
        return guarded(userId, sessionId, "get", () -> delegate.get(userId, sessionId, key, stateType));
    }

    @Override
    public <T extends State> List<T> getList(
            String userId, String sessionId, String key, Class<T> stateType) {
        if (isFrameworkAnonymousListGet(userId, sessionId, key)) {
            return List.of();
        }
        return guarded(
                userId, sessionId, "getList", () -> delegate.getList(userId, sessionId, key, stateType));
    }

    @Override
    public boolean exists(String userId, String sessionId) {
        return guarded(userId, sessionId, "exists", () -> delegate.exists(userId, sessionId));
    }

    @Override
    public void delete(String userId, String sessionId) {
        guarded(userId, sessionId, "delete", () -> delegate.delete(userId, sessionId));
    }

    @Override
    public void delete(String userId, String sessionId, String key) {
        guarded(userId, sessionId, "deleteKey", () -> delegate.delete(userId, sessionId, key));
    }

    @Override
    public Set<String> listSessionIds(String userId) {
        StateStoreSlot slot = StateStoreSlot.allSessions(userId);
        return guarded(slot, "listSessionIds", () -> delegate.listSessionIds(userId));
    }

    @Override
    public void close() {
        guarded(StateStoreSlot.storeLifecycle(), "close", delegate::close);
    }

    private void guarded(String userId, String sessionId, String operation, Runnable action) {
        guarded(new StateStoreSlot(userId, sessionId), operation, action);
    }

    private void guarded(StateStoreSlot slot, String operation, Runnable action) {
        guarded(slot, operation, () -> {
            action.run();
            return null;
        });
    }

    private <T> T guarded(
            String userId, String sessionId, String operation, Supplier<T> action) {
        return guarded(new StateStoreSlot(userId, sessionId), operation, action);
    }

    private <T> T guarded(StateStoreSlot slot, String operation, Supplier<T> action) {
        try {
            return action.get();
        } catch (RuntimeException failure) {
            throw failures.record(slot, operation, failure);
        }
    }

    private boolean isFrameworkAnonymousGet(String userId, String sessionId, String key) {
        return isFrameworkFallbackSlot(userId, sessionId)
                && ("agent_state".equals(key) || "toolkit_activeGroups".equals(key));
    }

    private boolean isFrameworkAnonymousListGet(String userId, String sessionId, String key) {
        return isFrameworkFallbackSlot(userId, sessionId) && "memory_messages".equals(key);
    }

    private boolean isFrameworkFallbackSlot(String userId, String sessionId) {
        return userId == null && sessionId != null && !sessionId.isBlank();
    }
}
