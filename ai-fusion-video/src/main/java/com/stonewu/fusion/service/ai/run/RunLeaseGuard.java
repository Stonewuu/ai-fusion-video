package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.ExecutionStopReason;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Objects;

/** Database-time owner fencing for heartbeats and side-effect guards. */
@Service
public final class RunLeaseGuard {

    private final AgentRunRepository runs;
    private final OwnedExecutionRegistry executions;
    private final OwnedCancellationHandler ownedCancellations;
    private final AgentRuntimeSchedulers schedulers;

    public RunLeaseGuard(
            AgentRunRepository runs,
            OwnedExecutionRegistry executions,
            OwnedCancellationHandler ownedCancellations,
            AgentRuntimeSchedulers schedulers) {
        this.runs = Objects.requireNonNull(runs, "runs must not be null");
        this.executions = Objects.requireNonNull(
                executions, "executions must not be null");
        this.ownedCancellations = Objects.requireNonNull(
                ownedCancellations, "ownedCancellations must not be null");
        this.schedulers = Objects.requireNonNull(
                schedulers, "schedulers must not be null");
    }

    public Mono<Void> assertLease(
            String runId, String ownerInstanceId, long ownerEpoch) {
        OwnerIdentity owner = owner(runId, ownerInstanceId, ownerEpoch);
        return journal(() -> runs.hasValidOwnedLease(
                        owner.runId(), owner.ownerInstanceId(), owner.ownerEpoch()))
                .flatMap(valid -> valid
                        ? Mono.empty()
                        : Mono.error(new OwnerFencedException(owner.runId())));
    }

    public Mono<Void> heartbeat(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            Duration ownerLease) {
        OwnerIdentity owner = owner(runId, ownerInstanceId, ownerEpoch);
        requirePositive(ownerLease, "ownerLease");
        return journal(() -> runs.renewOwnedLease(
                        owner.runId(),
                        owner.ownerInstanceId(),
                        owner.ownerEpoch(),
                        ownerLease))
                .flatMap(renewed -> renewed
                        ? Mono.empty()
                        : classifyLostHeartbeat(owner))
                .onErrorResume(failure -> executions.interruptOwned(
                                owner.runId(),
                                owner.ownerInstanceId(),
                                owner.ownerEpoch(),
                                ExecutionStopReason.OWNER_FENCED)
                        .then(Mono.error(failure)));
    }

    public Mono<Void> observeCancellationSignal(
            String runId, String ownerInstanceId, long ownerEpoch) {
        OwnerIdentity owner = owner(runId, ownerInstanceId, ownerEpoch);
        return ownedCancellations.acknowledgeAndFinalize(
                        owner.runId(), owner.ownerInstanceId(), owner.ownerEpoch())
                .then();
    }

    private Mono<Void> classifyLostHeartbeat(OwnerIdentity owner) {
        return journal(() -> java.util.Optional.ofNullable(
                        runs.findRun(owner.runId())))
                .flatMap(current -> current.isPresent()
                                && AgentRunStatus.CANCEL_REQUESTED.name()
                                        .equals(current.get().getStatus())
                                && Objects.equals(
                                        current.get().getOwnerInstanceId(),
                                        owner.ownerInstanceId())
                                && Objects.equals(
                                        current.get().getOwnerEpoch(), owner.ownerEpoch())
                        ? ownedCancellations.acknowledgeAndFinalize(
                                        owner.runId(),
                                        owner.ownerInstanceId(),
                                        owner.ownerEpoch())
                                .then()
                        : executions.interruptOwned(
                                        owner.runId(),
                                        owner.ownerInstanceId(),
                                        owner.ownerEpoch(),
                                        ExecutionStopReason.OWNER_FENCED)
                                .then());
    }

    private <T> Mono<T> journal(java.util.concurrent.Callable<T> operation) {
        return Mono.fromCallable(operation).subscribeOn(schedulers.journal());
    }

    private OwnerIdentity owner(
            String runId, String ownerInstanceId, long ownerEpoch) {
        if (runId == null || runId.isBlank() || runId.length() > 64) {
            throw new IllegalArgumentException("runId must be 1 to 64 characters");
        }
        if (ownerInstanceId == null
                || ownerInstanceId.isBlank()
                || ownerInstanceId.length() > 128) {
            throw new IllegalArgumentException(
                    "ownerInstanceId must be 1 to 128 characters");
        }
        if (ownerEpoch <= 0) {
            throw new IllegalArgumentException("ownerEpoch must be positive");
        }
        return new OwnerIdentity(runId, ownerInstanceId, ownerEpoch);
    }

    private void requirePositive(Duration value, String field) {
        Objects.requireNonNull(value, field + " must not be null");
        if (value.isZero() || value.isNegative()) {
            throw new IllegalArgumentException(field + " must be greater than zero");
        }
    }

    private record OwnerIdentity(
            String runId, String ownerInstanceId, long ownerEpoch) {
    }

    public static final class OwnerFencedException extends RuntimeException {

        public OwnerFencedException(String runId) {
            super("Agent run owner is no longer valid: " + runId);
        }
    }
}
