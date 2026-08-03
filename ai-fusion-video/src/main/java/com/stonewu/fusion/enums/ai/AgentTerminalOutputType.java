package com.stonewu.fusion.enums.ai;

/** Stable legacy projection type emitted by a terminal run event. */
public enum AgentTerminalOutputType {

    DONE(AgentRunStatus.COMPLETED),
    ERROR(AgentRunStatus.FAILED),
    CANCELLED(AgentRunStatus.CANCELLED);

    private final AgentRunStatus terminalStatus;

    AgentTerminalOutputType(AgentRunStatus terminalStatus) {
        this.terminalStatus = terminalStatus;
    }

    public AgentRunStatus terminalStatus() {
        return terminalStatus;
    }

    public String getCode() {
        return name();
    }
}
