package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.service.ai.run.model.NormalizedModelUsage;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;
import java.util.regex.Pattern;

/** Default no-billing adapter backed by the already durable audit ledger. */
public final class AuditLedgerModelUsageSettlementAdapter
        implements ModelUsageSettlementPort {

    private static final Pattern IDEMPOTENCY_KEY = Pattern.compile(
            "[^:\\s]{1,64}:[^:\\s]{1,64}");

    @Override
    public Mono<String> settle(
            String idempotencyKey, NormalizedModelUsage usage) {
        return Mono.fromSupplier(() -> {
            if (idempotencyKey == null
                    || !IDEMPOTENCY_KEY.matcher(idempotencyKey).matches()) {
                throw new IllegalArgumentException(
                        "idempotency key must be runId:modelCallId");
            }
            Objects.requireNonNull(usage, "usage must not be null");
            byte[] material = ("afv-usage-v1:" + idempotencyKey)
                    .getBytes(StandardCharsets.UTF_8);
            return "usage-" + HexFormat.of().formatHex(sha256(material));
        });
    }

    private byte[] sha256(byte[] value) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(value);
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }
}
