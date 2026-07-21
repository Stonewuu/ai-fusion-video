package com.stonewu.fusion.service.ai.agentscope.kernel;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;

public record AgentKernelToolManifest(
        String toolName,
        String schemaSha256,
        boolean readOnly,
        boolean concurrencySafe) {

    public AgentKernelToolManifest {
        toolName = requireText(toolName, "toolName");
        if (toolName.indexOf('|') >= 0 || toolName.indexOf('\n') >= 0 || toolName.indexOf('\r') >= 0) {
            throw new IllegalArgumentException("toolName contains a reserved manifest delimiter");
        }
        schemaSha256 = requireText(schemaSha256, "schemaSha256").toLowerCase();
        if (!schemaSha256.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("schemaSha256 must be a SHA-256 hex string");
        }
    }

    public static String schemaSha256(String schema) {
        return sha256(Objects.requireNonNull(schema, "schema must not be null"));
    }

    static String sha256(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(value.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
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
