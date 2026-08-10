package com.stonewu.fusion.service.ai.run.model;

/** Provider-neutral token usage persisted for one model call. */
public record NormalizedModelUsage(
        Long inputTokens,
        Long outputTokens,
        Long reasoningTokens,
        Long cacheTokens,
        String usageJson) {

    private static final int MAX_USAGE_JSON_CHARS = 262_144;

    public NormalizedModelUsage {
        requireNonNegative(inputTokens, "inputTokens");
        requireNonNegative(outputTokens, "outputTokens");
        requireNonNegative(reasoningTokens, "reasoningTokens");
        requireNonNegative(cacheTokens, "cacheTokens");
        if (usageJson != null) {
            if (usageJson.isBlank()) {
                throw new IllegalArgumentException(
                        "usageJson must not be blank when present");
            }
            if (usageJson.length() > MAX_USAGE_JSON_CHARS) {
                throw new IllegalArgumentException(
                        "usageJson exceeds the maximum supported size");
            }
        }
    }

    public static NormalizedModelUsage tokens(long inputTokens, long outputTokens) {
        return new NormalizedModelUsage(
                inputTokens, outputTokens, null, null, null);
    }

    private static void requireNonNegative(Long value, String field) {
        if (value != null && value < 0) {
            throw new IllegalArgumentException(field + " must not be negative");
        }
    }
}
