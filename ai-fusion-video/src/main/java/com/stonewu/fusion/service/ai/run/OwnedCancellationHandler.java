package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.ExecutionStopReason;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.Objects;

/** Owner-side cancellation acknowledgement, interruption, and durable convergence. */
@Component
public final class OwnedCancellationHandler {

    private final AgentRunRepository runs;
    private final OwnedExecutionRegistry executions;
    private final AgentRunReconciliationService reconciliation;
    private final AgentRuntimeSchedulers schedulers;

    public OwnedCancellationHandler(
            AgentRunRepository runs,
            OwnedExecutionRegistry executions,
            AgentRunReconciliationService reconciliation,
            AgentRuntimeSchedulers schedulers) {
        this.runs = Objects.requireNonNull(runs, "runs must not be null");
        this.executions = Objects.requireNonNull(
                executions, "executions must not be null");
        this.reconciliation = Objects.requireNonNull(
                reconciliation, "reconciliation must not be null");
        this.schedulers = Objects.requireNonNull(
                schedulers, "schedulers must not be null");
    }

    public Mono<Boolean> acknowledgeAndFinalize(
            String runId,
            String ownerInstanceId,
            long ownerEpoch) {
        String safeRunId = requireText(runId, "runId");
        String safeOwnerInstanceId = requireText(
                ownerInstanceId, "ownerInstanceId");
        if (ownerEpoch <= 0) {
            return Mono.error(new IllegalArgumentException(
                    "ownerEpoch must be positive"));
        }
        return journal(() -> runs.acknowledgeOwnedCancellation(
                        safeRunId, safeOwnerInstanceId, ownerEpoch))
                .flatMap(acknowledged -> acknowledged
                        ? executions.interruptOwned(
                                        safeRunId,
                                        safeOwnerInstanceId,
                                        ownerEpoch,
                                        ExecutionStopReason.CANCEL_REQUESTED)
                                .then(reconciliation.reconcileExpiredOwner(safeRunId))
                                .thenReturn(true)
                        : Mono.just(false));
    }

    private <T> Mono<T> journal(java.util.concurrent.Callable<T> operation) {
        return Mono.fromCallable(operation).subscribeOn(schedulers.journal());
    }

    private String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }
}
