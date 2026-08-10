package com.stonewu.fusion.service.ai.agentscope.context;

import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;

import java.util.Objects;

/** Exact AgentScope tool policy selected for the current conversation. */
public record ToolPermissionContext(ToolExecutionMode mode) {

    public ToolPermissionContext {
        mode = Objects.requireNonNull(mode, "mode must not be null");
    }
}
