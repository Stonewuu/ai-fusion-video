package com.stonewu.fusion.service.ai.agentscope.state;

import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public final class InMemoryStateStoreFailureGuard implements StateStoreFailureGuard {

    private final ConcurrentMap<StateStoreSlot, StateStoreFailure> failures = new ConcurrentHashMap<>();

    @Override
    public void clear(StateStoreSlot slot) {
        failures.remove(requireSlot(slot));
    }

    @Override
    public StateStoreFailure record(StateStoreSlot slot, String operation, Throwable cause) {
        StateStoreSlot safeSlot = requireSlot(slot);
        StateStoreFailure candidate = existingForSlot(safeSlot, operation, cause);
        StateStoreFailure stored = failures.putIfAbsent(safeSlot, candidate);
        return stored != null ? stored : candidate;
    }

    @Override
    public Optional<StateStoreFailure> failure(StateStoreSlot slot) {
        return Optional.ofNullable(failures.get(requireSlot(slot)));
    }

    @Override
    public void throwIfFailed(StateStoreSlot slot) throws StateStoreFailure {
        StateStoreFailure stored = failures.get(requireSlot(slot));
        if (stored != null) {
            throw stored;
        }
    }

    private StateStoreFailure existingForSlot(
            StateStoreSlot slot, String operation, Throwable cause) {
        if (operation == null || operation.isBlank()) {
            throw new IllegalArgumentException("operation must not be blank");
        }
        Objects.requireNonNull(cause, "cause must not be null");
        if (cause instanceof StateStoreFailure stateStoreFailure
                && stateStoreFailure.slot().equals(slot)) {
            return stateStoreFailure;
        }
        return new StateStoreFailure(slot, operation, cause);
    }

    private StateStoreSlot requireSlot(StateStoreSlot slot) {
        return Objects.requireNonNull(slot, "slot must not be null");
    }
}
