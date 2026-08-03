package com.stonewu.fusion.repository.ai;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.TextNode;
import com.stonewu.fusion.entity.ai.AgentModelCallUsage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentModelCallStatus;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.mapper.ai.AgentModelCallUsageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.run.AgentEventEnvelopeSanitizer;
import com.stonewu.fusion.service.ai.run.model.ModelCallIdentityConflictException;
import com.stonewu.fusion.service.ai.run.model.NormalizedModelUsage;
import lombok.RequiredArgsConstructor;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** MySQL implementation of the call-level usage and settlement claim ledger. */
@Repository
@RequiredArgsConstructor
public class MybatisAgentModelCallUsageRepository
        implements AgentModelCallUsageRepository {

    private static final String SETTLEMENT_PENDING = "PENDING";
    private static final String SETTLEMENT_CLAIMED = "CLAIMED";
    private static final int MAX_CLAIM_BATCH = 500;
    private static final int MAX_CLAIM_OWNER_LENGTH = 128;
    private static final int MAX_DOWNSTREAM_ID_LENGTH = 128;
    private static final int MAX_SETTLEMENT_ERROR_LENGTH = 1024;
    private static final Duration MAX_CLAIM_LEASE = Duration.ofHours(24);
    private static final Set<AgentRunStatus> TERMINAL_RUN_STATUSES = Set.of(
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED);

    private final AgentModelCallUsageMapper usageMapper;
    private final AgentRunMapper runMapper;
    private final ObjectMapper objectMapper;
    private final AgentEventEnvelopeSanitizer sanitizer;

    @Override
    @Transactional
    public void startCall(
            String runId,
            String modelCallId,
            String provider,
            String modelCode) {
        String safeRunId = requireText(runId, "runId", 64);
        String safeCallId = requireText(modelCallId, "modelCallId", 64);
        String safeProvider = requireText(provider, "provider", 64);
        String safeModelCode = requireText(modelCode, "modelCode", 128);

        AgentRun run = runMapper.selectByRunIdForUpdate(safeRunId);
        if (run == null) {
            throw new IllegalArgumentException("Agent run does not exist: " + safeRunId);
        }
        AgentModelCallUsage existing = usageMapper.selectByRunAndCall(
                safeRunId, safeCallId);
        if (existing != null) {
            requireSameIdentity(
                    existing, safeProvider, safeModelCode, safeRunId, safeCallId);
            return;
        }
        if (!AgentRunStatus.RUNNING.name().equals(run.getStatus())) {
            throw new IllegalStateException(
                    "Agent run is not accepting new model calls: " + safeRunId);
        }

        LocalDateTime databaseNow = runMapper.selectDatabaseNow();
        AgentModelCallUsage started = AgentModelCallUsage.builder()
                .runId(safeRunId)
                .modelCallId(safeCallId)
                .provider(safeProvider)
                .modelCode(safeModelCode)
                .status(AgentModelCallStatus.STARTED.name())
                .settlementStatus(SETTLEMENT_PENDING)
                .settlementAttempts(0)
                .startedAt(databaseNow)
                .build();
        try {
            if (usageMapper.insert(started) != 1) {
                throw new IllegalStateException(
                        "Model-call usage insert did not affect exactly one row");
            }
        } catch (DuplicateKeyException duplicate) {
            AgentModelCallUsage raced = usageMapper.selectByRunAndCall(
                    safeRunId, safeCallId);
            if (raced == null) {
                throw duplicate;
            }
            requireSameIdentity(
                    raced, safeProvider, safeModelCode, safeRunId, safeCallId);
        }
    }

    @Override
    public boolean completeCall(
            String runId,
            String modelCallId,
            NormalizedModelUsage usage) {
        String safeRunId = requireText(runId, "runId", 64);
        String safeCallId = requireText(modelCallId, "modelCallId", 64);
        NormalizedModelUsage safeUsage = Objects.requireNonNull(
                usage, "usage must not be null");
        String safeUsageJson = sanitizeUsageJson(safeUsage.usageJson());
        LocalDateTime databaseNow = runMapper.selectDatabaseNow();
        return usageMapper.completeStarted(
                safeRunId,
                safeCallId,
                safeUsage.inputTokens(),
                safeUsage.outputTokens(),
                safeUsage.reasoningTokens(),
                safeUsage.cacheTokens(),
                safeUsageJson,
                databaseNow) == 1;
    }

    @Override
    public boolean failCall(
            String runId,
            String modelCallId,
            AgentModelCallStatus status) {
        String safeRunId = requireText(runId, "runId", 64);
        String safeCallId = requireText(modelCallId, "modelCallId", 64);
        AgentModelCallStatus terminal = Objects.requireNonNull(
                status, "status must not be null");
        if (terminal != AgentModelCallStatus.FAILED
                && terminal != AgentModelCallStatus.CANCELLED) {
            throw new IllegalArgumentException(
                    "failCall only accepts FAILED or CANCELLED");
        }
        return usageMapper.finishStarted(
                safeRunId,
                safeCallId,
                terminal.name(),
                runMapper.selectDatabaseNow()) == 1;
    }

    @Override
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public List<AgentModelCallUsage> claimSettlementBatch(
            String claimOwner,
            Duration claimLease,
            int limit) {
        String safeOwner = requireText(
                claimOwner, "claimOwner", MAX_CLAIM_OWNER_LENGTH);
        Duration safeLease = requireClaimLease(claimLease);
        if (limit <= 0 || limit > MAX_CLAIM_BATCH) {
            throw new IllegalArgumentException(
                    "limit must be between 1 and " + MAX_CLAIM_BATCH);
        }

        LocalDateTime databaseNow = runMapper.selectDatabaseNow();
        LocalDateTime claimUntil = databaseNow.plus(safeLease)
                .truncatedTo(ChronoUnit.MILLIS);
        List<AgentModelCallUsage> candidates =
                usageMapper.selectSettlementCandidatesForUpdate(databaseNow, limit);
        for (AgentModelCallUsage candidate : candidates) {
            if (usageMapper.claimCandidate(
                    candidate.getId(), safeOwner, claimUntil, databaseNow) != 1) {
                throw new IllegalStateException(
                        "Settlement claim lost while the candidate row was locked: "
                                + candidate.getId());
            }
            candidate.setSettlementStatus(SETTLEMENT_CLAIMED);
            candidate.setSettlementClaimOwner(safeOwner);
            candidate.setSettlementClaimUntil(claimUntil);
            candidate.setSettlementAttempts(candidate.getSettlementAttempts() + 1);
            candidate.setNextSettlementAttemptAt(null);
        }
        return List.copyOf(candidates);
    }

    @Override
    public boolean markSettled(
            long usageId,
            String claimOwner,
            String downstreamSettlementId) {
        requirePositiveId(usageId);
        String safeOwner = requireText(
                claimOwner, "claimOwner", MAX_CLAIM_OWNER_LENGTH);
        String safeDownstreamId = requireText(
                downstreamSettlementId,
                "downstreamSettlementId",
                MAX_DOWNSTREAM_ID_LENGTH);
        return usageMapper.markSettled(
                usageId, safeOwner, safeDownstreamId) == 1;
    }

    @Override
    public boolean releaseSettlementForRetry(
            long usageId,
            String claimOwner,
            Instant nextAttemptAt,
            String sanitizedError) {
        requirePositiveId(usageId);
        String safeOwner = requireText(
                claimOwner, "claimOwner", MAX_CLAIM_OWNER_LENGTH);
        LocalDateTime retryAt = toDatabaseTime(Objects.requireNonNull(
                nextAttemptAt, "nextAttemptAt must not be null"));
        String safeError = sanitizeSettlementError(sanitizedError);
        return usageMapper.releaseSettlementForRetry(
                usageId, safeOwner, retryAt, safeError) == 1;
    }

    @Override
    @Transactional
    public boolean markRunUsageSettledIfAllCallsSettled(String runId) {
        String safeRunId = requireText(runId, "runId", 64);
        AgentRun run = runMapper.selectByRunIdForUpdate(safeRunId);
        if (run == null) {
            return false;
        }
        AgentRunStatus status = parseRunStatus(run.getStatus());
        if (!TERMINAL_RUN_STATUSES.contains(status)) {
            return false;
        }
        return usageMapper.markRunUsageSettledIfNoUnsettledCall(
                safeRunId, runMapper.selectDatabaseNow()) == 1;
    }

    private void requireSameIdentity(
            AgentModelCallUsage existing,
            String provider,
            String modelCode,
            String runId,
            String modelCallId) {
        if (Objects.equals(existing.getProvider(), provider)
                && Objects.equals(existing.getModelCode(), modelCode)) {
            return;
        }
        throw new ModelCallIdentityConflictException(
                runId,
                modelCallId,
                existing.getProvider(),
                existing.getModelCode(),
                provider,
                modelCode);
    }

    private String sanitizeUsageJson(String usageJson) {
        if (usageJson == null) {
            return null;
        }
        try {
            JsonNode parsed = objectMapper.readTree(usageJson);
            if (parsed == null || !parsed.isObject()) {
                throw new IllegalArgumentException(
                        "usageJson must contain one JSON object");
            }
            return objectMapper.writeValueAsString(sanitizer.sanitize(parsed));
        } catch (JsonProcessingException invalidJson) {
            throw new IllegalArgumentException(
                    "usageJson must be valid JSON", invalidJson);
        }
    }

    private String sanitizeSettlementError(String error) {
        String value = error == null || error.isBlank()
                ? "Settlement failed"
                : sanitizer.sanitize(TextNode.valueOf(error)).asText();
        return value.length() <= MAX_SETTLEMENT_ERROR_LENGTH
                ? value
                : value.substring(0, MAX_SETTLEMENT_ERROR_LENGTH);
    }

    private AgentRunStatus parseRunStatus(String status) {
        try {
            return AgentRunStatus.valueOf(status);
        } catch (RuntimeException invalid) {
            throw new IllegalStateException(
                    "Agent run has an invalid status: " + status, invalid);
        }
    }

    private Duration requireClaimLease(Duration value) {
        Objects.requireNonNull(value, "claimLease must not be null");
        if (value.isZero() || value.isNegative() || value.compareTo(MAX_CLAIM_LEASE) > 0) {
            throw new IllegalArgumentException(
                    "claimLease must be positive and no longer than "
                            + MAX_CLAIM_LEASE);
        }
        return value;
    }

    private String requireText(String value, String field, int maxLength) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        if (value.length() > maxLength) {
            throw new IllegalArgumentException(
                    field + " exceeds maximum length " + maxLength);
        }
        if (!StandardCharsets.US_ASCII.newEncoder().canEncode(value)) {
            throw new IllegalArgumentException(field + " must contain only ASCII characters");
        }
        return value;
    }

    private void requirePositiveId(long usageId) {
        if (usageId <= 0) {
            throw new IllegalArgumentException("usageId must be positive");
        }
    }

    private LocalDateTime toDatabaseTime(Instant value) {
        return LocalDateTime.ofInstant(
                value.truncatedTo(ChronoUnit.MILLIS), ZoneOffset.UTC);
    }
}
