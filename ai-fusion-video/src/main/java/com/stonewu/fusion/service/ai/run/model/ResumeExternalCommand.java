package com.stonewu.fusion.service.ai.run.model;

import java.time.Duration;
import java.util.Objects;

public record ResumeExternalCommand(
        String runId,
        long currentUserId,
        String internalExecutorId,
        String toolCallId,
        String resultPayloadJson,
        String newOwnerInstanceId,
        Duration ownerLease) {

    public ResumeExternalCommand {
        runId = requireText(runId, "runId");
        if (currentUserId <= 0) {
            throw new IllegalArgumentException("currentUserId must be positive");
        }
        internalExecutorId = requireText(internalExecutorId, "internalExecutorId");
        toolCallId = requireText(toolCallId, "toolCallId");
        resultPayloadJson = requireText(resultPayloadJson, "resultPayloadJson");
        newOwnerInstanceId = requireText(newOwnerInstanceId, "newOwnerInstanceId");
        Objects.requireNonNull(ownerLease, "ownerLease must not be null");
        if (ownerLease.isZero() || ownerLease.isNegative()) {
            throw new IllegalArgumentException("ownerLease must be greater than zero");
        }
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }
}
