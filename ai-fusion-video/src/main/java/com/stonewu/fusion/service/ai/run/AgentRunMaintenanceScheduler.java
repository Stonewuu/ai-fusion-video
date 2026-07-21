package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.config.AgentScopeV2Properties;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/** Periodic lease heartbeat, cancellation retry, and expired-owner convergence. */
@Component
@Slf4j
public final class AgentRunMaintenanceScheduler {

    private static final int BATCH_SIZE = 100;

    private final OwnedExecutionRegistry executions;
    private final RunLeaseGuard leases;
    private final AgentRunReconciliationService reconciliation;
    private final CancellationCoordinator cancellations;
    private final Duration ownerLease;
    private final AtomicBoolean running = new AtomicBoolean();

    public AgentRunMaintenanceScheduler(
            OwnedExecutionRegistry executions,
            RunLeaseGuard leases,
            AgentRunReconciliationService reconciliation,
            CancellationCoordinator cancellations,
            AgentScopeV2Properties properties) {
        this.executions = Objects.requireNonNull(
                executions, "executions must not be null");
        this.leases = Objects.requireNonNull(leases, "leases must not be null");
        this.reconciliation = Objects.requireNonNull(
                reconciliation, "reconciliation must not be null");
        this.cancellations = Objects.requireNonNull(
                cancellations, "cancellations must not be null");
        this.ownerLease = Objects.requireNonNull(
                properties, "properties must not be null")
                .getExecution()
                .getOwnerLease();
    }

    @Scheduled(
            initialDelayString =
                    "${fusion.agentscope.v2.execution.maintenance-initial-delay-ms:10000}",
            fixedDelayString =
                    "${fusion.agentscope.v2.execution.maintenance-delay-ms:5000}")
    public void maintain() {
        if (!running.compareAndSet(false, true)) {
            return;
        }
        maintainOnce()
                .doFinally(ignored -> running.set(false))
                .subscribe(
                        ignored -> { },
                        failure -> log.error(
                                "Agent run maintenance failed: type={}",
                                failure.getClass().getSimpleName()));
    }

    public Mono<Void> maintainOnce() {
        return heartbeatOwned()
                .then(reconciliation.reconcileBatch(BATCH_SIZE))
                .then(cancellations.retryBatch(BATCH_SIZE));
    }

    private Mono<Void> heartbeatOwned() {
        return Flux.fromIterable(executions.snapshot())
                .concatMap(handle -> leases.heartbeat(
                                handle.runId(),
                                handle.ownerInstanceId(),
                                handle.ownerEpoch(),
                                ownerLease)
                        .onErrorResume(failure -> {
                            log.warn(
                                    "Agent owner heartbeat failed and was fenced: runId={}, type={}",
                                    handle.runId(),
                                    failure.getClass().getSimpleName());
                            return Mono.empty();
                        }))
                .then();
    }
}
