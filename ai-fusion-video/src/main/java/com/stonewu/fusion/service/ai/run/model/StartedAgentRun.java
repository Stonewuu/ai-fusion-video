package com.stonewu.fusion.service.ai.run.model;

import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;

import java.time.Instant;
import java.util.Objects;

/** Persisted identity and ownership returned after a run admission transaction commits. */
public record StartedAgentRun(
        String runId,
        String conversationId,
        String agentStateSessionId,
        String ownerInstanceId,
        long ownerEpoch,
        Instant leaseUntil,
        Instant deadline,
        AgentKernelSnapshot kernelSnapshot,
        long initialMessageOrder) {

    public StartedAgentRun {
        requireText(runId, "runId");
        requireText(conversationId, "conversationId");
        requireText(agentStateSessionId, "agentStateSessionId");
        requireText(ownerInstanceId, "ownerInstanceId");
        if (ownerEpoch <= 0) {
            throw new IllegalArgumentException("ownerEpoch must be positive");
        }
        Objects.requireNonNull(leaseUntil, "leaseUntil must not be null");
        Objects.requireNonNull(deadline, "deadline must not be null");
        Objects.requireNonNull(kernelSnapshot, "kernelSnapshot must not be null");
        if (initialMessageOrder <= 0) {
            throw new IllegalArgumentException("initialMessageOrder must be positive");
        }
    }

    private static void requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
    }
}
