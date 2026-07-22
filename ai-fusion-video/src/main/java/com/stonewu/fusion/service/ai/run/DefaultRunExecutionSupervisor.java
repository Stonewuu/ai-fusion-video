package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentRuntimeErrorCode;
import com.stonewu.fusion.enums.ai.AgentTerminalOutputType;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextRequest;
import com.stonewu.fusion.service.ai.agentscope.context.ParentAgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.kernel.HarnessLeaseCache;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.kernel.RunConfigUnavailableException;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import com.stonewu.fusion.service.ai.run.model.ExecutionStopReason;
import com.stonewu.fusion.service.ai.run.model.ResumeAgentExecutionCommand;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.StartAgentExecutionCommand;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

@Service
public final class DefaultRunExecutionSupervisor implements RunExecutionSupervisor {

    private final AgentExecutionFactory executionFactory;
    private final OwnedExecutionRegistry executions;
    private final AgentEventChunkCoalescer chunks;
    private final AgentEventJournal journal;
    private final RunTerminalCoordinator terminals;
    private final AgentMessageProjectionService projections;
    private final HarnessLeaseCache kernelLeaseCache;
    private final AgentKernelSnapshotBuilder snapshotBuilder;
    private final RunShutdownCancellationPort shutdownCancellation;
    private final RunLeaseGuard leases;
    private final AgentRunRedisSignalService signals;
    private final AgentEventEnvelopeSanitizer sanitizer;
    private final AtomicBoolean accepting = new AtomicBoolean(true);
    private final AtomicReference<Mono<Void>> shutdownSignal = new AtomicReference<>();
    private AgentRuntimeMetrics metrics = AgentRuntimeMetrics.noop();

    @Autowired
    void setMetrics(AgentRuntimeMetrics metrics) {
        this.metrics = Objects.requireNonNull(metrics, "metrics must not be null");
    }

    public DefaultRunExecutionSupervisor(
            AgentExecutionFactory executionFactory,
            OwnedExecutionRegistry executions,
            AgentEventChunkCoalescer chunks,
            AgentEventJournal journal,
            RunTerminalCoordinator terminals,
            AgentMessageProjectionService projections,
            HarnessLeaseCache kernelLeaseCache,
            AgentKernelSnapshotBuilder snapshotBuilder,
            RunShutdownCancellationPort shutdownCancellation,
            RunLeaseGuard leases,
            AgentRunRedisSignalService signals,
            AgentEventEnvelopeSanitizer sanitizer) {
        this.executionFactory = Objects.requireNonNull(
                executionFactory, "executionFactory must not be null");
        this.executions = Objects.requireNonNull(executions, "executions must not be null");
        this.chunks = Objects.requireNonNull(chunks, "chunks must not be null");
        this.journal = Objects.requireNonNull(journal, "journal must not be null");
        this.terminals = Objects.requireNonNull(terminals, "terminals must not be null");
        this.projections = Objects.requireNonNull(
                projections, "projections must not be null");
        this.kernelLeaseCache = Objects.requireNonNull(
                kernelLeaseCache, "kernelLeaseCache must not be null");
        this.snapshotBuilder = Objects.requireNonNull(
                snapshotBuilder, "snapshotBuilder must not be null");
        this.shutdownCancellation = Objects.requireNonNull(
                shutdownCancellation, "shutdownCancellation must not be null");
        this.leases = Objects.requireNonNull(leases, "leases must not be null");
        this.signals = Objects.requireNonNull(signals, "signals must not be null");
        this.sanitizer = Objects.requireNonNull(sanitizer, "sanitizer must not be null");
    }

    @Override
    public Mono<Void> start(StartAgentExecutionCommand command) {
        return Mono.defer(() -> {
            requireAccepting();
            StartAgentExecutionCommand safeCommand = Objects.requireNonNull(
                    command, "command must not be null");
            requireStartSnapshot(safeCommand);
            return startResolved(
                    safeCommand.run().runId(),
                    safeCommand.run().ownerInstanceId(),
                    safeCommand.run().ownerEpoch(),
                    safeCommand.run().agentStateSessionId(),
                    safeCommand.run().deadline(),
                    safeCommand.messages(),
                    safeCommand.kernelSpec(),
                    safeCommand.runtimeContextRequest());
        });
    }

    @Override
    public Mono<Void> resume(ResumeAgentExecutionCommand command) {
        return Mono.defer(() -> {
            requireAccepting();
            ResumeAgentExecutionCommand safeCommand = Objects.requireNonNull(
                    command, "command must not be null");
            requireResumeSnapshot(safeCommand);
            return executionFactory.resolve(safeCommand.kernelSnapshot())
                    .flatMap(spec -> startResolved(
                            safeCommand.run().runId(),
                            safeCommand.run().newOwnerInstanceId(),
                            safeCommand.run().newOwnerEpoch(),
                            safeCommand.run().sessionId(),
                            safeCommand.run().deadline(),
                            safeCommand.messages(),
                            spec,
                            safeCommand.runtimeContextRequest()))
                    .onErrorResume(RunConfigUnavailableException.class, failure ->
                            terminalFailure(
                                    safeCommand.run().runId(),
                                    safeCommand.run().newOwnerInstanceId(),
                                    safeCommand.run().newOwnerEpoch(),
                                    safeCommand.runtimeContextRequest(),
                                    safeCommand.run().sessionId(),
                                    AgentRuntimeErrorCode.RUN_CONFIG_UNAVAILABLE,
                                    failure.getMessage()));
        });
    }

    private Mono<Void> startResolved(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            String stateSessionId,
            Instant deadline,
            java.util.List<io.agentscope.core.message.Msg> messages,
            AgentKernelSpec spec,
            AgentScopeRuntimeContextRequest runtimeRequest) {
        requireRuntimeIdentity(
                runId, ownerInstanceId, ownerEpoch,
                stateSessionId, deadline, runtimeRequest);
        return executionFactory.start(
                        runId,
                        ownerInstanceId,
                        ownerEpoch,
                        stateSessionId,
                        messages,
                        spec,
                        runtimeRequest,
                        deadline)
                .flatMap(execution -> executions.registerAndLaunch(
                        execution,
                        deadline,
                        chunks::coalesce,
                        events -> events.concatMap(
                                        event -> appendOwned(execution, event), 1)
                                .then(),
                        ignored -> signals.cancellations(runId)
                                .next()
                                .flatMap(cancelledRunId ->
                                        leases.observeCancellationSignal(
                                                cancelledRunId,
                                                ownerInstanceId,
                                                ownerEpoch))
                                .onErrorResume(failure -> Mono.empty()),
                        outcome -> completeExecution(execution, outcome)))
                .onErrorResume(
                        OwnedExecutionRegistry.ExecutionAlreadyOwnedException.class,
                        Mono::error)
                .onErrorResume(failure -> terminalFailure(
                                runId,
                                ownerInstanceId,
                                ownerEpoch,
                                runtimeRequest,
                                stateSessionId,
                                classifyStartFailure(failure),
                                failure.getMessage())
                        .then(Mono.error(failure)));
    }

    private Mono<Void> appendOwned(
            AgentExecution execution,
            AgentEventEnvelope event) {
        return appendRequired(
                        execution.runId(),
                        execution.ownerInstanceId(),
                        execution.ownerEpoch(),
                        event,
                        false)
                .flatMap(committed -> mirrorToParent(execution, committed));
    }

    private Mono<CommittedAgentEvent> appendRequired(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            AgentEventEnvelope event,
            boolean parentWrite) {
        return journal.appendOwned(runId, ownerInstanceId, ownerEpoch, event)
                .flatMap(committed -> committed
                        .map(Mono::just)
                        .orElseGet(() -> Mono.error(parentWrite
                                ? new ParentOwnerLostException(runId)
                                : new OwnerLostException(runId))));
    }

    private Mono<Void> mirrorToParent(
            AgentExecution execution,
            CommittedAgentEvent childEvent) {
        ParentAgentRunContext parent = execution.parentRun();
        if (parent == null) {
            return Mono.empty();
        }
        return appendRequired(
                        parent.runId(),
                        parent.ownerInstanceId(),
                        parent.ownerEpoch(),
                        mirroredChildEvent(parent, childEvent),
                        true)
                .then();
    }

    private AgentEventEnvelope mirroredChildEvent(
            ParentAgentRunContext parent,
            CommittedAgentEvent childEvent) {
        AgentEventEnvelope original = childEvent.envelope();
        JsonNode originalPayload = original.payload();
        if (!originalPayload.isObject()) {
            throw new IllegalStateException(
                    "Child Agent event payload must be a JSON object: "
                            + original.rawEventId());
        }
        ObjectNode payload = (ObjectNode) originalPayload;
        payload.put("_platformMirroredChildEvent", true);
        payload.put("childRunId", childEvent.runId());
        payload.put("childSequence", childEvent.sequence());
        String mirrorMaterial = childEvent.runId() + '\0' + original.rawEventId();
        String mirroredRawEventId = "child-" + UUID.nameUUIDFromBytes(
                        mirrorMaterial.getBytes(StandardCharsets.UTF_8))
                .toString()
                .replace("-", "");
        return new AgentEventEnvelope(
                mirroredRawEventId,
                original.rawEventType(),
                "main/" + parent.agentName(),
                original.replyId(),
                original.blockId(),
                original.toolCallId(),
                parent.toolCallId(),
                parent.agentName(),
                original.outputType(),
                payload,
                original.createdAt());
    }

    private Mono<Void> completeExecution(
            AgentExecution execution,
            AgentExecutionHandle.Outcome outcome) {
        metrics.providerClosed();
        AgentRunStatus providerStatus = switch (outcome.kind()) {
            case COMPLETED -> AgentRunStatus.COMPLETED;
            case INTERRUPTED -> outcome.stopReason() == ExecutionStopReason.CANCEL_REQUESTED
                    ? AgentRunStatus.CANCELLED
                    : AgentRunStatus.FAILED;
            case OVERFLOW, SOURCE_FAILURE, JOURNAL_FAILURE -> AgentRunStatus.FAILED;
        };
        metrics.providerTerminal(providerStatus);
        return switch (outcome.kind()) {
            case COMPLETED -> terminalCompleted(execution);
            case OVERFLOW -> terminalFailure(
                    execution,
                    AgentRuntimeErrorCode.AGENT_EVENT_BACKPRESSURE_OVERFLOW,
                    "Agent event ingress exceeded its bounded capacity");
            case SOURCE_FAILURE -> terminalFailure(
                    execution,
                    AgentRuntimeErrorCode.AGENTSCOPE_INTERNAL_ERROR,
                    outcome.failure().getMessage());
            case JOURNAL_FAILURE -> outcome.failure() instanceof OwnerLostException
                    ? Mono.empty()
                    : terminalFailure(
                            execution,
                            AgentRuntimeErrorCode.EVENT_PERSIST_FAILED,
                            outcome.failure().getMessage());
            case INTERRUPTED -> outcome.stopReason() == ExecutionStopReason.DEADLINE
                    ? terminalFailure(
                            execution,
                            AgentRuntimeErrorCode.MODEL_TIMEOUT,
                            "Agent run deadline expired")
                    : Mono.empty();
        };
    }

    @Override
    public Mono<Boolean> interruptOwned(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            ExecutionStopReason reason) {
        return executions.interruptOwned(
                runId, ownerInstanceId, ownerEpoch, reason);
    }

    @Override
    public Mono<Void> shutdown(Duration drainTimeout) {
        Objects.requireNonNull(drainTimeout, "drainTimeout must not be null");
        if (drainTimeout.isZero() || drainTimeout.isNegative()) {
            return Mono.error(new IllegalArgumentException(
                    "drainTimeout must be greater than zero"));
        }
        Mono<Void> existing = shutdownSignal.get();
        if (existing != null) {
            return existing;
        }
        Mono<Void> candidate = Mono.defer(() -> {
                    accepting.set(false);
                    return executions.awaitEmpty(drainTimeout)
                            .onErrorResume(TimeoutException.class, ignored ->
                                    Flux.fromIterable(executions.snapshot())
                                            .concatMap(handle -> shutdownCancellation
                                                    .request(handle.runId())
                                                    .then(handle.interrupt(
                                                            ExecutionStopReason.SHUTDOWN))
                                                    .onErrorResume(failure -> Mono.empty()))
                                            .then())
                            .then(kernelLeaseCache.drainAndClose(drainTimeout));
                })
                .cache();
        return shutdownSignal.compareAndSet(null, candidate)
                ? candidate
                : shutdownSignal.get();
    }

    private Mono<Void> terminalCompleted(AgentExecution execution) {
        return terminals.terminateOwned(
                        terminalRequest(
                                execution.runId(),
                                execution.userId(),
                                execution.stateSessionId(),
                                AgentRunStatus.COMPLETED,
                                AgentTerminalOutputType.DONE,
                                null,
                                null,
                                execution.parentToolCallId(),
                                execution.agentName()),
                        execution.ownerInstanceId(),
                        execution.ownerEpoch())
                .flatMap(this::projectTerminal);
    }

    private Mono<Void> terminalFailure(
            AgentExecution execution,
            AgentRuntimeErrorCode errorCode,
            String errorMessage) {
        return terminalFailure(
                execution.runId(),
                execution.ownerInstanceId(),
                execution.ownerEpoch(),
                execution.userId(),
                execution.stateSessionId(),
                execution.parentToolCallId(),
                execution.agentName(),
                errorCode,
                errorMessage);
    }

    private Mono<Void> terminalFailure(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            AgentScopeRuntimeContextRequest runtimeRequest,
            String stateSessionId,
            AgentRuntimeErrorCode errorCode,
            String errorMessage) {
        return terminalFailure(
                runId,
                ownerInstanceId,
                ownerEpoch,
                runtimeRequest.authenticatedUser().userId(),
                stateSessionId,
                runtimeRequest.run().parentToolCallId(),
                runtimeRequest.run().agentName(),
                errorCode,
                errorMessage);
    }

    private Mono<Void> terminalFailure(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            long userId,
            String stateSessionId,
            String parentToolCallId,
            String agentName,
            AgentRuntimeErrorCode errorCode,
            String errorMessage) {
        return terminals.terminateOwned(
                        terminalRequest(
                                runId,
                                userId,
                                stateSessionId,
                                AgentRunStatus.FAILED,
                                AgentTerminalOutputType.ERROR,
                                errorCode,
                                sanitizeMessage(errorMessage),
                                parentToolCallId,
                                agentName),
                        ownerInstanceId,
                        ownerEpoch)
                .flatMap(this::projectTerminal);
    }

    private RunTerminalRequest terminalRequest(
            String runId,
            long userId,
            String stateSessionId,
            AgentRunStatus status,
            AgentTerminalOutputType outputType,
            AgentRuntimeErrorCode errorCode,
            String errorMessage,
            String parentToolCallId,
            String agentName) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", outputType.name())
                .put("finished", true);
        if (errorCode != null) {
            payload.put("errorCode", errorCode.name());
            payload.put("error", errorMessage);
        }
        AgentEventEnvelope envelope = new AgentEventEnvelope(
                "terminal-" + UUID.randomUUID().toString().replace("-", ""),
                "RUN_TERMINAL",
                agentName == null ? "main" : "main/" + agentName,
                null,
                null,
                null,
                parentToolCallId,
                agentName,
                outputType.name(),
                sanitizer.sanitize(payload),
                Instant.now());
        return new RunTerminalRequest(
                runId,
                new StateStoreSlot(String.valueOf(userId), stateSessionId),
                Set.of(AgentRunStatus.RUNNING),
                status,
                outputType,
                errorCode,
                errorMessage,
                envelope);
    }

    private Mono<Void> projectTerminal(
            Optional<com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent> committed) {
        return committed
                .map(event -> projections.projectThrough(
                        event.runId(), event.sequence()))
                .orElseGet(Mono::empty);
    }

    private AgentRuntimeErrorCode classifyStartFailure(Throwable failure) {
        return failure instanceof RunConfigUnavailableException
                ? AgentRuntimeErrorCode.RUN_CONFIG_UNAVAILABLE
                : AgentRuntimeErrorCode.AGENTSCOPE_INTERNAL_ERROR;
    }

    private String sanitizeMessage(String message) {
        String value = message == null || message.isBlank()
                ? "Agent execution failed"
                : message;
        JsonNode sanitized = sanitizer.sanitize(
                JsonNodeFactory.instance.objectNode().put("message", value));
        String safe = sanitized.path("message").asText("Agent execution failed");
        return safe.length() <= 1024 ? safe : safe.substring(0, 1024);
    }

    private void requireStartSnapshot(StartAgentExecutionCommand command) {
        if (!sameSnapshot(command.run().kernelSnapshot(), command.kernelSnapshot())) {
            throw new IllegalArgumentException(
                    "Start execution snapshot does not match the persisted run snapshot");
        }
        requireSpecMatchesSnapshot(command.kernelSpec(), command.kernelSnapshot());
    }

    private void requireResumeSnapshot(ResumeAgentExecutionCommand command) {
        if (!Objects.equals(command.run().kernelFingerprint(),
                    command.kernelSnapshot().fingerprint())
                || !Objects.equals(command.run().agentDefinitionSnapshotJson(),
                    command.kernelSnapshot().snapshotJson())) {
            throw new IllegalArgumentException(
                    "Resume execution snapshot does not match the persisted run snapshot");
        }
    }

    private boolean sameSnapshot(AgentKernelSnapshot left, AgentKernelSnapshot right) {
        return Objects.equals(left.fingerprint(), right.fingerprint())
                && Objects.equals(left.snapshotJson(), right.snapshotJson());
    }

    private void requireSpecMatchesSnapshot(
            AgentKernelSpec spec, AgentKernelSnapshot snapshot) {
        AgentKernelSnapshot rebuilt = snapshotBuilder.build(spec);
        if (!sameSnapshot(rebuilt, snapshot)) {
            throw new IllegalArgumentException(
                    "Live Agent kernel does not match the persisted immutable snapshot");
        }
    }

    private void requireRuntimeIdentity(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            String stateSessionId,
            Instant deadline,
            AgentScopeRuntimeContextRequest runtimeRequest) {
        AgentRunContext run = runtimeRequest.run();
        if (!Objects.equals(runId, run.runId())
                || !Objects.equals(ownerInstanceId, run.ownerInstanceId())
                || ownerEpoch != run.ownerEpoch()
                || !Objects.equals(
                        stateSessionId,
                        runtimeRequest.conversation().agentStateSessionId())
                || !Objects.equals(deadline, run.deadline())) {
            throw new IllegalArgumentException(
                    "RuntimeContext run identity does not match the execution command");
        }
    }

    private void requireAccepting() {
        if (!accepting.get()) {
            throw new IllegalStateException("Agent runtime is shutting down");
        }
    }

    static final class OwnerLostException extends RuntimeException {
        private OwnerLostException(String runId) {
            super("Agent run ownership was lost: " + runId);
        }
    }

    static final class ParentOwnerLostException extends RuntimeException {
        private ParentOwnerLostException(String runId) {
            super("Parent Agent run ownership was lost: " + runId);
        }
    }
}
