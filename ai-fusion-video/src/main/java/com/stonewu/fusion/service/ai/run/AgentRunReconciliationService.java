package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentRuntimeErrorCode;
import com.stonewu.fusion.enums.ai.AgentTerminalOutputType;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.ExecutionStopReason;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.SystemTerminalActor;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;

/** Converges expired owner leases without ever re-executing a run. */
@Service
public final class AgentRunReconciliationService {

    private static final String OWNER_LOST_MESSAGE =
            "Agent run owner lease expired";
    private static final String CANCEL_MESSAGE = "Agent run was cancelled";

    private final AgentRunRepository runs;
    private final RunTerminalCoordinator terminals;
    private final OwnedExecutionRegistry executions;
    private final AgentMessageProjectionService projections;
    private final AgentEventEnvelopeSanitizer sanitizer;
    private final AgentRuntimeSchedulers schedulers;
    private AgentRuntimeMetrics metrics = AgentRuntimeMetrics.noop();

    @Autowired
    void setMetrics(AgentRuntimeMetrics metrics) {
        this.metrics = Objects.requireNonNull(metrics, "metrics must not be null");
    }

    public AgentRunReconciliationService(
            AgentRunRepository runs,
            RunTerminalCoordinator terminals,
            OwnedExecutionRegistry executions,
            AgentMessageProjectionService projections,
            AgentEventEnvelopeSanitizer sanitizer,
            AgentRuntimeSchedulers schedulers) {
        this.runs = Objects.requireNonNull(runs, "runs must not be null");
        this.terminals = Objects.requireNonNull(terminals, "terminals must not be null");
        this.executions = Objects.requireNonNull(
                executions, "executions must not be null");
        this.projections = Objects.requireNonNull(
                projections, "projections must not be null");
        this.sanitizer = Objects.requireNonNull(sanitizer, "sanitizer must not be null");
        this.schedulers = Objects.requireNonNull(
                schedulers, "schedulers must not be null");
    }

    public Mono<Void> reconcileExpiredOwner(String runId) {
        String safeRunId = requireRunId(runId);
        return journal(() -> runs.findRun(safeRunId))
                .flatMap(run -> run == null ? Mono.empty() : reconcile(run));
    }

    public Mono<Void> reconcileBatch(int limit) {
        return journal(() -> runs.findExpiredLeaseCandidates(limit))
                .flatMapMany(Flux::fromIterable)
                .concatMap(run -> reconcileExpiredOwner(run.getRunId()))
                .then();
    }

    private Mono<Void> reconcile(AgentRun run) {
        AgentRunStatus status = status(run);
        RunTerminalRequest request;
        ExecutionStopReason stopReason;
        if (status == AgentRunStatus.RUNNING) {
            request = ownerLostRequest(run);
            stopReason = ExecutionStopReason.OWNER_FENCED;
        } else if (status == AgentRunStatus.CANCEL_REQUESTED) {
            request = cancelledRequest(run);
            stopReason = ExecutionStopReason.CANCEL_REQUESTED;
        } else {
            return Mono.empty();
        }
        return terminals.terminateSystem(request, SystemTerminalActor.OWNER_RECONCILER)
                .flatMap(committed -> {
                    if (committed.isPresent() && status == AgentRunStatus.RUNNING) {
                        metrics.ownerLost();
                    }
                    Mono<Void> interrupt = committed.isPresent() && hasOwner(run)
                        ? executions.interruptOwned(
                                        run.getRunId(),
                                        run.getOwnerInstanceId(),
                                        run.getOwnerEpoch(),
                                        stopReason)
                                .then()
                        : Mono.empty();
                    Mono<Void> projection = committed
                            .map(event -> projections.projectThrough(
                                    event.runId(), event.sequence()))
                            .orElseGet(Mono::empty);
                    return interrupt.then(projection);
                });
    }

    private RunTerminalRequest ownerLostRequest(AgentRun run) {
        return terminalRequest(
                run,
                Set.of(AgentRunStatus.RUNNING),
                AgentRunStatus.FAILED,
                AgentTerminalOutputType.ERROR,
                AgentRuntimeErrorCode.OWNER_LOST,
                OWNER_LOST_MESSAGE);
    }

    private RunTerminalRequest cancelledRequest(AgentRun run) {
        return terminalRequest(
                run,
                Set.of(AgentRunStatus.CANCEL_REQUESTED),
                AgentRunStatus.CANCELLED,
                AgentTerminalOutputType.CANCELLED,
                AgentRuntimeErrorCode.RUN_CANCELLED,
                CANCEL_MESSAGE);
    }

    private RunTerminalRequest terminalRequest(
            AgentRun run,
            Set<AgentRunStatus> expected,
            AgentRunStatus terminalStatus,
            AgentTerminalOutputType outputType,
            AgentRuntimeErrorCode errorCode,
            String message) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", outputType.name())
                .put("errorCode", errorCode.name())
                .put("error", message)
                .put("finished", true);
        AgentEventEnvelope envelope = new AgentEventEnvelope(
                "reconciled-" + UUID.randomUUID().toString().replace("-", ""),
                "RUN_TERMINAL",
                run.getParentRunId() == null
                        ? "main"
                        : "main/" + run.getAgentName(),
                null,
                null,
                null,
                run.getParentToolCallId(),
                run.getAgentName(),
                outputType.name(),
                sanitizer.sanitize(payload),
                Instant.now());
        return new RunTerminalRequest(
                run.getRunId(),
                new StateStoreSlot(
                        String.valueOf(run.getUserId()),
                        run.getAgentStateSessionId()),
                expected,
                terminalStatus,
                outputType,
                errorCode,
                message,
                envelope);
    }

    private AgentRunStatus status(AgentRun run) {
        try {
            return AgentRunStatus.valueOf(run.getStatus());
        } catch (RuntimeException invalid) {
            throw new IllegalStateException(
                    "Agent run has an unsupported status: " + run.getStatus(), invalid);
        }
    }

    private boolean hasOwner(AgentRun run) {
        return run.getOwnerInstanceId() != null
                && !run.getOwnerInstanceId().isBlank()
                && run.getOwnerEpoch() != null
                && run.getOwnerEpoch() > 0;
    }

    private <T> Mono<T> journal(java.util.concurrent.Callable<T> operation) {
        return Mono.fromCallable(operation).subscribeOn(schedulers.journal());
    }

    private String requireRunId(String runId) {
        if (runId == null || runId.isBlank() || runId.length() > 64) {
            throw new IllegalArgumentException("runId must be 1 to 64 characters");
        }
        return runId;
    }
}
