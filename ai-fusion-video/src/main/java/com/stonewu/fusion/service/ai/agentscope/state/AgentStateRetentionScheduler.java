package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.AgentRuntimeInstanceIdentity;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicBoolean;

/** Polls the durable policy and runs one distributed AgentState cleanup job when due. */
@Component
@Slf4j
public final class AgentStateRetentionScheduler {

    private static final int BATCH_SIZE = 100;
    private static final int CLAIM_LEASE_MINUTES = 120;

    private final AgentStateCleanupPolicyService policies;
    private final AgentStateRetentionCleaner cleaner;
    private final AgentRuntimeInstanceIdentity instanceIdentity;
    private final AgentRuntimeSchedulers schedulers;
    private final AtomicBoolean running = new AtomicBoolean();

    public AgentStateRetentionScheduler(
            AgentStateCleanupPolicyService policies,
            AgentStateRetentionCleaner cleaner,
            AgentRuntimeInstanceIdentity instanceIdentity,
            AgentRuntimeSchedulers schedulers) {
        this.policies = Objects.requireNonNull(policies, "policies must not be null");
        this.cleaner = Objects.requireNonNull(cleaner, "cleaner must not be null");
        this.instanceIdentity = Objects.requireNonNull(
                instanceIdentity, "instanceIdentity must not be null");
        this.schedulers = Objects.requireNonNull(
                schedulers, "schedulers must not be null");
    }

    @Scheduled(
            initialDelayString =
                    "${fusion.agentscope.v2.state.cleanup-poll-initial-delay-ms:30000}",
            fixedDelayString =
                    "${fusion.agentscope.v2.state.cleanup-poll-delay-ms:60000}")
    public void maintain() {
        if (!running.compareAndSet(false, true)) {
            return;
        }
        maintainOnce()
                .doFinally(ignored -> running.set(false))
                .subscribe(
                        ignored -> { },
                        failure -> log.error(
                                "AgentState retention cleanup failed: type={}",
                                failure.getClass().getSimpleName()));
    }

    public Mono<Void> maintainOnce() {
        String owner = instanceIdentity.value() + ':'
                + UUID.randomUUID().toString().replace("-", "");
        return journal(() -> policies.tryClaim(
                        owner, CLAIM_LEASE_MINUTES))
                .flatMap(claim -> claim
                        .map(value -> runClaim(value, owner))
                        .orElseGet(Mono::empty));
    }

    private Mono<Void> runClaim(
            AgentStateCleanupPolicyService.CleanupClaim claim,
            String owner) {
        return cleaner.cleanExpired(claim.retentionDays(), BATCH_SIZE)
                .flatMap(cleaned -> journal(() -> {
                            policies.complete(owner);
                            return cleaned;
                        })
                        .doOnNext(count -> log.info(
                                "AgentState retention cleanup completed: conversations={}",
                                count)))
                .then()
                .onErrorResume(failure -> journal(() -> {
                            policies.release(owner);
                            return Optional.empty();
                        })
                        .onErrorResume(releaseFailure -> Mono.empty())
                        .then(Mono.error(failure)));
    }

    private <T> Mono<T> journal(Callable<T> operation) {
        return Mono.fromCallable(operation).subscribeOn(schedulers.journal());
    }
}
