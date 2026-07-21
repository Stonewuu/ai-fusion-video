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
import org.springframework.transaction.support.TransactionTemplate;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.Instant;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;

/** Durably requests cancellation before broadcasting or interrupting executions. */
@Service
public final class CancellationCoordinator implements RunShutdownCancellationPort {

    private static final Duration CANCEL_RETRY_DELAY = Duration.ofSeconds(5);
    private static final String CANCEL_MESSAGE = "Agent run was cancelled";

    private final AgentRunRepository runs;
    private final AgentRunRedisSignalService signals;
    private final OwnedExecutionRegistry executions;
    private final RunTerminalCoordinator terminals;
    private final AgentEventEnvelopeSanitizer sanitizer;
    private final TransactionTemplate transactions;
    private final AgentRuntimeSchedulers schedulers;

    public CancellationCoordinator(
            AgentRunRepository runs,
            AgentRunRedisSignalService signals,
            OwnedExecutionRegistry executions,
            RunTerminalCoordinator terminals,
            AgentEventEnvelopeSanitizer sanitizer,
            TransactionTemplate transactions,
            AgentRuntimeSchedulers schedulers) {
        this.runs = Objects.requireNonNull(runs, "runs must not be null");
        this.signals = Objects.requireNonNull(signals, "signals must not be null");
        this.executions = Objects.requireNonNull(
                executions, "executions must not be null");
        this.terminals = Objects.requireNonNull(terminals, "terminals must not be null");
        this.sanitizer = Objects.requireNonNull(sanitizer, "sanitizer must not be null");
        this.transactions = Objects.requireNonNull(
                transactions, "transactions must not be null");
        this.schedulers = Objects.requireNonNull(
                schedulers, "schedulers must not be null");
    }

    public Mono<AgentRunStatus> cancel(String runId, long currentUserId) {
        String safeRunId = requireRunId(runId);
        if (currentUserId <= 0) {
            return Mono.error(new IllegalArgumentException(
                    "currentUserId must be positive"));
        }
        return requestAuthorized(safeRunId, currentUserId)
                .flatMap(tree -> publishAndInterrupt(tree.affected())
                        .then(currentStatus(tree.root().getRunId())));
    }

    @Override
    public Mono<Void> request(String runId) {
        return requestInternal(requireRunId(runId))
                .flatMap(tree -> publishAndInterrupt(tree.affected()));
    }

    public Mono<Void> cancelChildren(String parentRunId) {
        String safeParentRunId = requireRunId(parentRunId);
        return requestDescendants(safeParentRunId)
                .flatMap(tree -> publishAndInterrupt(tree.affected()));
    }

    public Mono<Void> retry(String runId) {
        return request(requireRunId(runId));
    }

    public Mono<Void> retryBatch(int limit) {
        return journal(() -> runs.findCancellationRetryCandidates(limit))
                .flatMapMany(Flux::fromIterable)
                .concatMap(run -> retry(run.getRunId()))
                .then();
    }

    private Mono<AgentRunRepository.CancellationTree> requestAuthorized(
            String runId, long currentUserId) {
        return transaction(() -> runs.requestAuthorizedCancellationTree(
                runId, currentUserId));
    }

    private Mono<AgentRunRepository.CancellationTree> requestInternal(String runId) {
        return transaction(() -> runs.requestInternalCancellationTree(runId));
    }

    private Mono<AgentRunRepository.CancellationTree> requestDescendants(
            String parentRunId) {
        return transaction(() -> runs.requestDescendantCancellationTree(parentRunId));
    }

    private Mono<Void> publishAndInterrupt(java.util.List<AgentRun> affected) {
        return Flux.fromIterable(affected)
                .concatMap(this::publishAndInterrupt)
                .then();
    }

    private Mono<Void> publishAndInterrupt(AgentRun run) {
        Mono<Void> publish = signals.publishCancel(run.getRunId())
                .then(journal(() -> runs.markCancellationBroadcast(
                        run.getRunId(), CANCEL_RETRY_DELAY)).then())
                .onErrorResume(ignored -> Mono.empty());
        if (!hasOwner(run)) {
            return publish.then(terminalCancelled(run));
        }
        return publish.then(executions.interruptOwned(
                        run.getRunId(),
                        run.getOwnerInstanceId(),
                        run.getOwnerEpoch(),
                        ExecutionStopReason.CANCEL_REQUESTED)
                .flatMap(interrupted -> interrupted
                        ? journal(() -> runs.acknowledgeOwnedCancellation(
                                        run.getRunId(),
                                        run.getOwnerInstanceId(),
                                        run.getOwnerEpoch()))
                                .then()
                        : Mono.empty()));
    }

    private Mono<Void> terminalCancelled(AgentRun run) {
        return terminals.terminateSystem(
                        cancelledRequest(run),
                        SystemTerminalActor.CANCELLATION_COORDINATOR)
                .then();
    }

    private Mono<AgentRunStatus> currentStatus(String runId) {
        return journal(() -> {
            AgentRun current = runs.findRun(runId);
            if (current == null) {
                throw new IllegalStateException(
                        "Cancelled Agent run disappeared: " + runId);
            }
            try {
                return AgentRunStatus.valueOf(current.getStatus());
            } catch (IllegalArgumentException invalid) {
                throw new IllegalStateException(
                        "Agent run has an unsupported status: " + current.getStatus(),
                        invalid);
            }
        });
    }

    private RunTerminalRequest cancelledRequest(AgentRun run) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", AgentTerminalOutputType.CANCELLED.name())
                .put("errorCode", AgentRuntimeErrorCode.RUN_CANCELLED.name())
                .put("error", CANCEL_MESSAGE)
                .put("finished", true);
        AgentEventEnvelope envelope = new AgentEventEnvelope(
                "cancelled-" + UUID.randomUUID().toString().replace("-", ""),
                "RUN_TERMINAL",
                run.getParentRunId() == null
                        ? "main"
                        : "main/" + run.getAgentName(),
                null,
                null,
                null,
                run.getParentToolCallId(),
                run.getAgentName(),
                AgentTerminalOutputType.CANCELLED.name(),
                sanitizer.sanitize(payload),
                Instant.now());
        return new RunTerminalRequest(
                run.getRunId(),
                new StateStoreSlot(
                        String.valueOf(run.getUserId()),
                        run.getAgentStateSessionId()),
                Set.of(AgentRunStatus.CANCEL_REQUESTED),
                AgentRunStatus.CANCELLED,
                AgentTerminalOutputType.CANCELLED,
                AgentRuntimeErrorCode.RUN_CANCELLED,
                CANCEL_MESSAGE,
                envelope);
    }

    private boolean hasOwner(AgentRun run) {
        return run.getOwnerInstanceId() != null
                && !run.getOwnerInstanceId().isBlank()
                && run.getOwnerEpoch() != null
                && run.getOwnerEpoch() > 0;
    }

    private <T> Mono<T> transaction(java.util.function.Supplier<T> operation) {
        return Mono.fromCallable(() -> Objects.requireNonNull(
                        transactions.execute(ignored -> operation.get()),
                        "Cancellation transaction returned no result"))
                .subscribeOn(schedulers.journal());
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
