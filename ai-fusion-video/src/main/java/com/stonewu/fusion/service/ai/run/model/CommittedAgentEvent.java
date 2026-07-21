package com.stonewu.fusion.service.ai.run.model;

import java.time.Instant;
import java.util.Objects;

/** Immutable view of an event after its per-run sequence is committed. */
public record CommittedAgentEvent(
        long eventId,
        String runId,
        long sequence,
        AgentEventEnvelope envelope,
        Instant committedAt) {

    public CommittedAgentEvent {
        if (eventId <= 0) {
            throw new IllegalArgumentException("eventId must be positive");
        }
        requireText(runId, "runId");
        if (sequence <= 0) {
            throw new IllegalArgumentException("sequence must be positive");
        }
        Objects.requireNonNull(envelope, "envelope must not be null");
        Objects.requireNonNull(committedAt, "committedAt must not be null");
    }

    public String outputType() {
        return envelope.outputType();
    }

    public com.fasterxml.jackson.databind.JsonNode projection() {
        return outputType() == null ? null : envelope.payload();
    }

    public boolean publishRequired() {
        return outputType() != null;
    }

    private static void requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
    }
}
