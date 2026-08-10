package com.stonewu.fusion.enums.ai;

/** Durable lifecycle states for one provider model call. */
public enum AgentModelCallStatus {

    STARTED(false),
    COMPLETED(true),
    FAILED(true),
    CANCELLED(true);

    private final boolean terminal;

    AgentModelCallStatus(boolean terminal) {
        this.terminal = terminal;
    }

    public boolean isTerminal() {
        return terminal;
    }

    public String getCode() {
        return name();
    }
}
