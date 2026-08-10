package com.stonewu.fusion.service.ai.run.model;

import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;

import java.time.Duration;
import java.time.Instant;
import java.util.Objects;

/**
 * Immutable child-admission request containing only caller-controlled child identity.
 * Parent-authoritative conversation, user, project, and session values are intentionally absent.
 */
public record StartChildAgentRunCommand(
        String childRunId,
        String parentRunId,
        String parentToolCallId,
        String parentOwnerInstanceId,
        long parentOwnerEpoch,
        String agentName,
        String agentDefinitionStableKey,
        AgentKernelSnapshot kernelSnapshot,
        String ownerInstanceId,
        Duration ownerLease,
        Instant deadline,
        String userContent,
        String referencesJson) {

    public StartChildAgentRunCommand {
        requireText(childRunId, "childRunId");
        requireText(parentRunId, "parentRunId");
        requireText(parentToolCallId, "parentToolCallId");
        requireText(parentOwnerInstanceId, "parentOwnerInstanceId");
        if (parentOwnerEpoch <= 0) {
            throw new IllegalArgumentException("parentOwnerEpoch must be positive");
        }
        requireText(agentName, "agentName");
        requireText(agentDefinitionStableKey, "agentDefinitionStableKey");
        Objects.requireNonNull(kernelSnapshot, "kernelSnapshot must not be null");
        requireText(ownerInstanceId, "ownerInstanceId");
        requirePositive(ownerLease, "ownerLease");
        Objects.requireNonNull(deadline, "deadline must not be null");
        requireText(userContent, "userContent");
        requireOptionalText(referencesJson, "referencesJson");
    }

    private static void requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
    }

    private static void requireOptionalText(String value, String field) {
        if (value != null && value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank when provided");
        }
    }

    private static void requirePositive(Duration value, String field) {
        Objects.requireNonNull(value, field + " must not be null");
        if (value.isZero() || value.isNegative()) {
            throw new IllegalArgumentException(field + " must be positive");
        }
    }
}
