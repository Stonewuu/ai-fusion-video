package com.stonewu.fusion.service.ai.agentscope.context;

import java.time.Instant;
import java.util.Objects;

public record AgentRunContext(
        String runId,
        String ownerInstanceId,
        long ownerEpoch,
        Instant deadline,
        String parentToolCallId,
        String agentName) {

    public AgentRunContext {
        runId = ContextValues.requireText(runId, "runId");
        ownerInstanceId = ContextValues.requireText(ownerInstanceId, "ownerInstanceId");
        ownerEpoch = ContextValues.requirePositive(ownerEpoch, "ownerEpoch");
        deadline = Objects.requireNonNull(deadline, "deadline must not be null");
        parentToolCallId = optionalText(parentToolCallId, "parentToolCallId");
        agentName = optionalText(agentName, "agentName");
        if ((parentToolCallId == null) != (agentName == null)) {
            throw new IllegalArgumentException(
                    "parentToolCallId and agentName must either both be present or both be absent");
        }
    }

    public AgentRunContext(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            Instant deadline) {
        this(runId, ownerInstanceId, ownerEpoch, deadline, null, null);
    }

    public boolean childRun() {
        return parentToolCallId != null;
    }

    private static String optionalText(String value, String field) {
        if (value != null && value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank when present");
        }
        return value;
    }
}
