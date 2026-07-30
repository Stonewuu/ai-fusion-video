package com.stonewu.fusion.service.ai.run.model;

import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public record ResumeConfirmationCommand(
        String runId,
        long currentUserId,
        String replyId,
        Set<String> decisionIds,
        Map<String, Boolean> decisionResults,
        String newOwnerInstanceId,
        Duration ownerLease) {

    public ResumeConfirmationCommand {
        runId = requireText(runId, "runId");
        if (currentUserId <= 0) {
            throw new IllegalArgumentException("currentUserId must be positive");
        }
        replyId = requireText(replyId, "replyId");
        decisionIds = Set.copyOf(Objects.requireNonNull(
                decisionIds, "decisionIds must not be null"));
        if (decisionIds.isEmpty()
                || decisionIds.stream().anyMatch(value -> value == null || value.isBlank())) {
            throw new IllegalArgumentException(
                    "decisionIds must contain only non-blank values");
        }
        decisionResults = Map.copyOf(Objects.requireNonNull(
                decisionResults, "decisionResults must not be null"));
        if (!decisionResults.keySet().equals(decisionIds)
                || decisionResults.values().stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException(
                    "decisionResults must contain one boolean for every decisionId");
        }
        newOwnerInstanceId = requireText(newOwnerInstanceId, "newOwnerInstanceId");
        requirePositive(ownerLease, "ownerLease");
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }

    private static void requirePositive(Duration value, String field) {
        Objects.requireNonNull(value, field + " must not be null");
        if (value.isZero() || value.isNegative()) {
            throw new IllegalArgumentException(field + " must be greater than zero");
        }
    }
}
