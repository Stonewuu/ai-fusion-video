package com.stonewu.fusion.service.ai.run.model;

import java.time.Instant;
import java.util.Objects;

public record PendingExternalExecution(
        String toolCallId,
        String toolName,
        String suspendedPayloadJson,
        Instant expiresAt) {

    public PendingExternalExecution {
        toolCallId = requireText(toolCallId, "toolCallId");
        toolName = requireText(toolName, "toolName");
        suspendedPayloadJson = requireText(
                suspendedPayloadJson, "suspendedPayloadJson");
        Objects.requireNonNull(expiresAt, "expiresAt must not be null");
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }
}
