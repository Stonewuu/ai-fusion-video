package com.stonewu.fusion.service.ai.run.model;

public record WaitingCheckpoint(
        String sessionId,
        String kernelFingerprint,
        String agentDefinitionSnapshotJson,
        long pausedThroughSequence) {

    public WaitingCheckpoint {
        sessionId = requireText(sessionId, "sessionId");
        if (kernelFingerprint == null || !kernelFingerprint.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException(
                    "kernelFingerprint must be lowercase SHA-256");
        }
        agentDefinitionSnapshotJson = requireText(
                agentDefinitionSnapshotJson, "agentDefinitionSnapshotJson");
        if (pausedThroughSequence < 0) {
            throw new IllegalArgumentException(
                    "pausedThroughSequence must not be negative");
        }
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }
}
