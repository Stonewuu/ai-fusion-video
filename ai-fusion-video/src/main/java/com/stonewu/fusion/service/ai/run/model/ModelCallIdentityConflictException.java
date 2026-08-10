package com.stonewu.fusion.service.ai.run.model;

/** Raised when a durable call id is retried with a different immutable identity. */
public final class ModelCallIdentityConflictException extends IllegalStateException {

    private final String runId;
    private final String modelCallId;
    private final String existingProvider;
    private final String existingModelCode;
    private final String requestedProvider;
    private final String requestedModelCode;

    public ModelCallIdentityConflictException(
            String runId,
            String modelCallId,
            String existingProvider,
            String existingModelCode,
            String requestedProvider,
            String requestedModelCode) {
        super("Model call identity conflicts with the durable usage ledger for runId="
                + runId + ", modelCallId=" + modelCallId);
        this.runId = runId;
        this.modelCallId = modelCallId;
        this.existingProvider = existingProvider;
        this.existingModelCode = existingModelCode;
        this.requestedProvider = requestedProvider;
        this.requestedModelCode = requestedModelCode;
    }

    public String getRunId() {
        return runId;
    }

    public String getModelCallId() {
        return modelCallId;
    }

    public String getExistingProvider() {
        return existingProvider;
    }

    public String getExistingModelCode() {
        return existingModelCode;
    }

    public String getRequestedProvider() {
        return requestedProvider;
    }

    public String getRequestedModelCode() {
        return requestedModelCode;
    }
}
