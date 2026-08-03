package com.stonewu.fusion.service.ai.run.model;

import java.time.Instant;
import java.util.Objects;
import java.util.Set;

public record PendingConfirmation(
        String replyId,
        Set<String> decisionIds,
        String pendingToolCallsJson,
        String suspendedToolCallsJson,
        Instant expiresAt) {

    public PendingConfirmation {
        replyId = requireText(replyId, "replyId");
        decisionIds = Set.copyOf(Objects.requireNonNull(
                decisionIds, "decisionIds must not be null"));
        if (decisionIds.isEmpty()) {
            throw new IllegalArgumentException("decisionIds must not be empty");
        }
        if (decisionIds.stream().anyMatch(value -> value == null || value.isBlank())) {
            throw new IllegalArgumentException("decisionIds must not contain blank values");
        }
        pendingToolCallsJson = requireText(
                pendingToolCallsJson, "pendingToolCallsJson");
        suspendedToolCallsJson = requireText(
                suspendedToolCallsJson, "suspendedToolCallsJson");
        Objects.requireNonNull(expiresAt, "expiresAt must not be null");
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }
}
