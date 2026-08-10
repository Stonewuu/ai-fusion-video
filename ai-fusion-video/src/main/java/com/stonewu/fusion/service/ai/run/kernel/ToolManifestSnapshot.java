package com.stonewu.fusion.service.ai.run.kernel;

import java.util.Locale;
import java.util.Objects;

public record ToolManifestSnapshot(
        String name,
        String schemaSha256,
        boolean readOnly,
        boolean concurrencySafe,
        String implementationVersion) {

    public ToolManifestSnapshot {
        name = requireText(name, "name");
        schemaSha256 = requireText(schemaSha256, "schemaSha256").toLowerCase(Locale.ROOT);
        if (!schemaSha256.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("schemaSha256 must be a SHA-256 hex string");
        }
        implementationVersion = requireText(implementationVersion, "implementationVersion");
    }

    static String requireText(String value, String field) {
        Objects.requireNonNull(value, field + " must not be null");
        String normalized = value.trim();
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return normalized;
    }
}
