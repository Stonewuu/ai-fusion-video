package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.node.TextNode;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.TransactionDefinition;
import org.springframework.transaction.support.TransactionTemplate;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/** Claims committed outbox rows briefly and publishes Redis wake-up hints. */
@Service
@RequiredArgsConstructor
@Slf4j
public class AgentEventOutboxPublisher {

    private static final int MAX_BATCH_SIZE = 500;
    private static final int MAX_OWNER_PREFIX_LENGTH = 95;
    private static final int MAX_ERROR_LENGTH = 1024;
    private static final Duration CLAIM_LEASE = Duration.ofSeconds(30);
    private static final long MIN_RETRY_MILLIS = 100;
    private static final long MAX_RETRY_MILLIS = 30_000;

    private final AgentEventMapper eventMapper;
    private final AgentRunMapper runMapper;
    private final PlatformTransactionManager transactionManager;
    private final AgentRunRedisSignalService signals;
    private final AgentEventEnvelopeSanitizer sanitizer;
    private final AgentRuntimeSchedulers schedulers;
    private AgentRuntimeMetrics metrics = AgentRuntimeMetrics.noop();

    @Autowired
    void setMetrics(AgentRuntimeMetrics metrics) {
        this.metrics = java.util.Objects.requireNonNull(
                metrics, "metrics must not be null");
    }

    public Mono<Void> publishBatch(String owner, int limit) {
        String claimToken = claimToken(owner);
        int safeLimit = requireLimit(limit);
        return Mono.fromCallable(() -> claim(claimToken, safeLimit))
                .subscribeOn(schedulers.journal())
                .flatMapMany(Flux::fromIterable)
                .concatMap(event -> publishOne(claimToken, event))
                .then(refreshBacklog());
    }

    private Mono<Void> publishOne(String claimToken, AgentEvent event) {
        return signals.publishWakeup(event.getRunId(), event.getSequenceNo())
                .then(markPublished(event.getId(), claimToken))
                .flatMap(marked -> marked
                        ? Mono.<Void>empty()
                        : Mono.error(new OutboxClaimLostException(event.getId())))
                .onErrorResume(failure -> releaseForRetry(
                                event, claimToken, failure)
                        .doOnNext(released -> logPublishFailure(
                                event.getId(), failure, released))
                        .onErrorResume(releaseFailure -> {
                            log.error(
                                    "Agent outbox retry release failed: eventId={}, type={}",
                                    event.getId(),
                                    releaseFailure.getClass().getSimpleName());
                            return Mono.empty();
                        })
                        .then());
    }

    private List<AgentEvent> claim(String claimToken, int limit) {
        TransactionTemplate transaction = new TransactionTemplate(transactionManager);
        transaction.setIsolationLevel(TransactionDefinition.ISOLATION_READ_COMMITTED);
        transaction.setPropagationBehavior(TransactionDefinition.PROPAGATION_REQUIRED);
        List<AgentEvent> claimed = transaction.execute(
                ignored -> claimTransaction(claimToken, limit));
        if (claimed == null) {
            throw new IllegalStateException("Agent outbox claim transaction returned no result");
        }
        return claimed;
    }

    private List<AgentEvent> claimTransaction(String claimToken, int limit) {
        LocalDateTime databaseNow = runMapper.selectDatabaseNow();
        List<AgentEvent> candidates = new ArrayList<>(limit);
        candidates.addAll(eventMapper.selectPendingPublishCandidatesForUpdate(
                databaseNow, limit));
        int remaining = limit - candidates.size();
        if (remaining > 0) {
            candidates.addAll(eventMapper.selectExpiredPublishCandidatesForUpdate(
                    databaseNow, remaining));
        }
        LocalDateTime claimUntil = databaseNow.plus(CLAIM_LEASE);
        for (AgentEvent event : candidates) {
            if (eventMapper.claimPublishCandidate(
                    event.getId(), claimToken, claimUntil, databaseNow) != 1) {
                throw new IllegalStateException(
                        "Agent outbox claim changed while its row was locked: "
                                + event.getId());
            }
            event.setPublishStatus("CLAIMED");
            event.setPublishClaimOwner(claimToken);
            event.setPublishClaimUntil(claimUntil);
            event.setPublishAttempts(event.getPublishAttempts() + 1);
            event.setNextPublishAttemptAt(null);
        }
        return List.copyOf(candidates);
    }

    private Mono<Boolean> markPublished(long eventId, String claimToken) {
        return Mono.fromCallable(() ->
                        eventMapper.markPublished(eventId, claimToken) == 1)
                .subscribeOn(schedulers.journal());
    }

    private Mono<Boolean> releaseForRetry(
            AgentEvent event,
            String claimToken,
            Throwable failure) {
        metrics.outboxRetry();
        return Mono.fromCallable(() -> {
                    LocalDateTime databaseNow = runMapper.selectDatabaseNow();
                    LocalDateTime nextAttempt = databaseNow.plusNanos(
                            retryDelayMillis(
                                    event.getId(), event.getPublishAttempts())
                                    * 1_000_000L);
                    return eventMapper.releasePublishForRetry(
                            event.getId(),
                            claimToken,
                            nextAttempt,
                            sanitizeFailure(failure)) == 1;
                })
                .subscribeOn(schedulers.journal());
    }

    private Mono<Void> refreshBacklog() {
        return Mono.fromCallable(eventMapper::countOutstandingPublish)
                .subscribeOn(schedulers.journal())
                .doOnNext(metrics::outboxBacklog)
                .then();
    }

    private long retryDelayMillis(long eventId, int attempts) {
        int exponent = Math.min(8, Math.max(0, attempts - 1));
        long base = Math.min(MAX_RETRY_MILLIS, MIN_RETRY_MILLIS << exponent);
        long jitterRange = Math.max(1, Math.min(base, MAX_RETRY_MILLIS - base + 1));
        long mixed = eventId * 0x9E3779B97F4A7C15L
                ^ Integer.toUnsignedLong(attempts * 0x85EBCA6B);
        long jitter = Math.floorMod(mixed, jitterRange);
        return Math.min(MAX_RETRY_MILLIS, base + jitter);
    }

    private String sanitizeFailure(Throwable failure) {
        String message = failure == null || failure.getMessage() == null
                || failure.getMessage().isBlank()
                ? failure == null ? "Outbox publish failed" : failure.getClass().getSimpleName()
                : failure.getMessage();
        String safe = sanitizer.sanitize(TextNode.valueOf(message)).asText();
        return safe.length() <= MAX_ERROR_LENGTH
                ? safe
                : safe.substring(0, MAX_ERROR_LENGTH);
    }

    private void logPublishFailure(long eventId, Throwable failure, boolean released) {
        log.warn(
                "Agent outbox publish failed: eventId={}, type={}, retryReleased={}",
                eventId,
                failure == null ? "unknown" : failure.getClass().getSimpleName(),
                released);
    }

    private String claimToken(String owner) {
        if (owner == null || owner.isBlank()) {
            throw new IllegalArgumentException("owner must not be blank");
        }
        if (owner.length() > MAX_OWNER_PREFIX_LENGTH
                || !StandardCharsets.US_ASCII.newEncoder().canEncode(owner)) {
            throw new IllegalArgumentException(
                    "owner must be at most " + MAX_OWNER_PREFIX_LENGTH
                            + " ASCII characters");
        }
        return owner + ':' + UUID.randomUUID().toString().replace("-", "");
    }

    private int requireLimit(int limit) {
        if (limit <= 0 || limit > MAX_BATCH_SIZE) {
            throw new IllegalArgumentException(
                    "limit must be between 1 and " + MAX_BATCH_SIZE);
        }
        return limit;
    }

    private static final class OutboxClaimLostException extends IllegalStateException {

        private OutboxClaimLostException(long eventId) {
            super("Agent outbox claim was lost for eventId=" + eventId);
        }
    }
}
