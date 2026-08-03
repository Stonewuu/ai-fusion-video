package com.stonewu.fusion.service.ai.run.model;

import java.time.Instant;
import java.util.Objects;

public record ResumedAgentRun(
        String runId,
        String conversationId,
        String sessionId,
        String kernelFingerprint,
        String agentDefinitionSnapshotJson,
        long pausedThroughSequence,
        String newOwnerInstanceId,
        long newOwnerEpoch,
        Instant leaseUntil,
        Instant deadline) {

    public ResumedAgentRun {
        requireText(runId, "runId");
        requireText(conversationId, "conversationId");
        requireText(sessionId, "sessionId");
        if (kernelFingerprint == null || !kernelFingerprint.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("kernelFingerprint must be lowercase SHA-256");
        }
        requireText(agentDefinitionSnapshotJson, "agentDefinitionSnapshotJson");
        if (pausedThroughSequence < 0) {
            throw new IllegalArgumentException("pausedThroughSequence must not be negative");
        }
        requireText(newOwnerInstanceId, "newOwnerInstanceId");
        if (newOwnerEpoch <= 0) {
            throw new IllegalArgumentException("newOwnerEpoch must be positive");
        }
        Objects.requireNonNull(leaseUntil, "leaseUntil must not be null");
        Objects.requireNonNull(deadline, "deadline must not be null");
    }

    private static void requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
    }
}
