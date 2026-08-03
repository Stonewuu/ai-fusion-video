package com.stonewu.fusion.service.ai.agentscope.state;

import java.util.Objects;

public final class StateStoreFailure extends RuntimeException {

    private final StateStoreSlot slot;
    private final String operation;

    public StateStoreFailure(StateStoreSlot slot, String operation, Throwable cause) {
        super(message(slot, operation), Objects.requireNonNull(cause, "cause must not be null"));
        this.slot = Objects.requireNonNull(slot, "slot must not be null");
        this.operation = requireOperation(operation);
    }

    public StateStoreSlot slot() {
        return slot;
    }

    public String operation() {
        return operation;
    }

    private static String message(StateStoreSlot slot, String operation) {
        return "AgentStateStore " + requireOperation(operation) + " failed for "
                + Objects.requireNonNull(slot, "slot must not be null");
    }

    private static String requireOperation(String operation) {
        if (operation == null || operation.isBlank()) {
            throw new IllegalArgumentException("operation must not be blank");
        }
        return operation.trim();
    }
}
