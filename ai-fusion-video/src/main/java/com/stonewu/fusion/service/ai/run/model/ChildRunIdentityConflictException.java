package com.stonewu.fusion.service.ai.run.model;

/** Raised when an idempotency identity points at a materially different child run. */
public final class ChildRunIdentityConflictException extends IllegalStateException {

    private final String parentRunId;
    private final String parentToolCallId;

    public ChildRunIdentityConflictException(String parentRunId, String parentToolCallId) {
        super(message(parentRunId, parentToolCallId));
        this.parentRunId = parentRunId;
        this.parentToolCallId = parentToolCallId;
    }

    public String getParentRunId() {
        return parentRunId;
    }

    public String getParentToolCallId() {
        return parentToolCallId;
    }

    private static String message(String parentRunId, String parentToolCallId) {
        requireText(parentRunId, "parentRunId");
        requireText(parentToolCallId, "parentToolCallId");
        return "Child run identity conflicts with the existing admission for parentRunId="
                + parentRunId + ", parentToolCallId=" + parentToolCallId;
    }

    private static void requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
    }
}
