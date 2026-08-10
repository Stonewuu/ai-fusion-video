package com.stonewu.fusion.service.ai.agentscope.tool;

import com.stonewu.fusion.enums.ai.AgentRunStatus;

import java.util.Objects;

public record PlatformSubAgentRun(
        String childRunId,
        String parentRunId,
        String parentToolCallId,
        String agentName,
        AgentRunStatus status,
        String result,
        String error) {

    public PlatformSubAgentRun {
        requireText(childRunId, "childRunId");
        requireText(parentRunId, "parentRunId");
        requireText(parentToolCallId, "parentToolCallId");
        requireText(agentName, "agentName");
        status = Objects.requireNonNull(status, "status must not be null");
        result = optionalText(result, "result");
        error = optionalText(error, "error");
    }

    public PlatformSubAgentRun(
            String childRunId,
            String parentRunId,
            String parentToolCallId,
            String agentName,
            AgentRunStatus status) {
        this(childRunId, parentRunId, parentToolCallId, agentName,
                status, null, null);
    }

    private static void requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
    }

    private static String optionalText(String value, String field) {
        if (value != null && value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank when present");
        }
        return value;
    }
}
