package com.stonewu.fusion.service.ai.run.model;

import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;

import java.time.Duration;
import java.time.Instant;
import java.util.Objects;

/**
 * Immutable request for atomically admitting a root agent run and its initial message.
 *
 * <p>The parent identity remains explicit so the coordinator can reject every non-root
 * admission through the root-only entry point. {@code stateSessionCandidate} is the
 * caller's identity for a new conversation; the coordinator owns the final generation
 * selection under the conversation lock.</p>
 */
public record StartAgentRunCommand(
        String runId,
        String conversationId,
        long userId,
        Long projectId,
        String agentType,
        String parentRunId,
        String parentToolCallId,
        String agentName,
        String stateSessionCandidate,
        AgentKernelSnapshot kernelSnapshot,
        String ownerInstanceId,
        Duration ownerLease,
        Instant deadline,
        String userContent,
        String referencesJson) {

    public StartAgentRunCommand {
        requireText(runId, "runId");
        requireText(conversationId, "conversationId");
        if (userId <= 0) {
            throw new IllegalArgumentException("userId must be positive");
        }
        if (projectId != null && projectId <= 0) {
            throw new IllegalArgumentException("projectId must be positive when provided");
        }
        requireText(agentType, "agentType");
        requireOptionalText(parentRunId, "parentRunId");
        requireOptionalText(parentToolCallId, "parentToolCallId");
        requireOptionalText(agentName, "agentName");
        requireText(stateSessionCandidate, "stateSessionCandidate");
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
