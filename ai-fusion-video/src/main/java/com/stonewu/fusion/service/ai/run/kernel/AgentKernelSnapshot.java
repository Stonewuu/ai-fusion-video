package com.stonewu.fusion.service.ai.run.kernel;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Locale;
import java.util.Objects;

public record AgentKernelSnapshot(
        AgentKernelSnapshotPayload payload,
        String canonicalJson,
        String fingerprint) {

    public AgentKernelSnapshot {
        payload = Objects.requireNonNull(payload, "payload must not be null");
        canonicalJson = Objects.requireNonNull(canonicalJson, "canonicalJson must not be null");
        if (canonicalJson.isBlank()) {
            throw new IllegalArgumentException("canonicalJson must not be blank");
        }
        fingerprint = Objects.requireNonNull(fingerprint, "fingerprint must not be null")
                .toLowerCase(Locale.ROOT);
        if (!fingerprint.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("fingerprint must be a SHA-256 hex string");
        }
        byte[] expected = sha256(canonicalJson.getBytes(StandardCharsets.UTF_8));
        byte[] supplied = HexFormat.of().parseHex(fingerprint);
        if (!MessageDigest.isEqual(expected, supplied)) {
            throw new IllegalArgumentException("fingerprint does not match canonicalJson");
        }
    }

    public byte[] canonicalBytes() {
        return canonicalJson.getBytes(StandardCharsets.UTF_8);
    }

    public String snapshotJson() {
        return canonicalJson;
    }

    static String fingerprint(byte[] canonicalBytes) {
        return HexFormat.of().formatHex(sha256(canonicalBytes));
    }

    static byte[] sha256(byte[] value) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(value);
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }
}
