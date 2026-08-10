package com.stonewu.fusion.service.ai.run.model;

import java.nio.charset.StandardCharsets;

/** Canonical durable SSE cursor. */
public record RunCursor(String runId, long afterSequence) {

    public RunCursor {
        if (runId == null || runId.isBlank()) {
            throw new IllegalArgumentException("runId must not be blank");
        }
        if (runId.length() > 64
                || !StandardCharsets.US_ASCII.newEncoder().canEncode(runId)) {
            throw new IllegalArgumentException(
                    "runId must be at most 64 ASCII characters");
        }
        if (afterSequence < 0) {
            throw new IllegalArgumentException(
                    "afterSequence must not be negative");
        }
    }
}
