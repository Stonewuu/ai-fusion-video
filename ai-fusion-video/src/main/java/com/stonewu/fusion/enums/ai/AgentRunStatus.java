package com.stonewu.fusion.enums.ai;

/** Durable states accepted by the agent-run persistence contract. */
public enum AgentRunStatus {

    RUNNING(false),
    WAITING_CONFIRMATION(false),
    WAITING_EXTERNAL(false),
    CANCEL_REQUESTED(false),
    COMPLETED(true),
    FAILED(true),
    CANCELLED(true);

    private final boolean terminal;

    AgentRunStatus(boolean terminal) {
        this.terminal = terminal;
    }

    public boolean isTerminal() {
        return terminal;
    }

    public boolean isActive() {
        return !terminal;
    }

    public String getCode() {
        return name();
    }
}
