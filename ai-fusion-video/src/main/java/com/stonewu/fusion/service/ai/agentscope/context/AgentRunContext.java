package com.stonewu.fusion.service.ai.agentscope.context;

import java.time.Instant;
import java.util.Objects;

public record AgentRunContext(
        String runId,
        String ownerInstanceId,
        long ownerEpoch,
        Instant deadline) {

    public AgentRunContext {
        runId = ContextValues.requireText(runId, "runId");
        ownerInstanceId = ContextValues.requireText(ownerInstanceId, "ownerInstanceId");
        ownerEpoch = ContextValues.requirePositive(ownerEpoch, "ownerEpoch");
        deadline = Objects.requireNonNull(deadline, "deadline must not be null");
    }
}
