package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.service.ai.agentscope.context.ParentAgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentCommand;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRun;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRunPort;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.ChildRunAdmission;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import com.stonewu.fusion.service.ai.run.model.StartAgentExecutionCommand;
import com.stonewu.fusion.service.ai.run.model.StartChildAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;

@Service
public final class PlatformSubAgentRunService implements PlatformSubAgentRunPort {

    private static final Set<String> TERMINAL_OUTPUT_TYPES =
            Set.of("DONE", "ERROR", "CANCELLED");

    private final AgentRunCoordinator coordinator;
    private final AgentKernelSnapshotBuilder snapshots;
    private final AgentInputHistoryMapper inputHistory;
    private final AgentExecutionRuntimeContextRequests runtimeContexts;
    private final RunExecutionSupervisor supervisor;
    private final AgentRunReplayService replay;
    private final AgentMessageProjectionService projections;
    private final CancellationCoordinator cancellations;
    private final AgentRuntimeInstanceIdentity instanceIdentity;
    private final Duration ownerLease;

    public PlatformSubAgentRunService(
            AgentRunCoordinator coordinator,
            AgentKernelSnapshotBuilder snapshots,
            AgentInputHistoryMapper inputHistory,
            AgentExecutionRuntimeContextRequests runtimeContexts,
            RunExecutionSupervisor supervisor,
            AgentRunReplayService replay,
            AgentMessageProjectionService projections,
            CancellationCoordinator cancellations,
            AgentRuntimeInstanceIdentity instanceIdentity,
            AgentScopeV2Properties properties) {
        this.coordinator = Objects.requireNonNull(coordinator, "coordinator must not be null");
        this.snapshots = Objects.requireNonNull(snapshots, "snapshots must not be null");
        this.inputHistory = Objects.requireNonNull(inputHistory, "inputHistory must not be null");
        this.runtimeContexts = Objects.requireNonNull(runtimeContexts, "runtimeContexts must not be null");
        this.supervisor = Objects.requireNonNull(supervisor, "supervisor must not be null");
        this.replay = Objects.requireNonNull(replay, "replay must not be null");
        this.projections = Objects.requireNonNull(
                projections, "projections must not be null");
        this.cancellations = Objects.requireNonNull(
                cancellations, "cancellations must not be null");
        this.instanceIdentity = Objects.requireNonNull(
                instanceIdentity, "instanceIdentity must not be null");
        this.ownerLease = Objects.requireNonNull(properties, "properties must not be null")
                .getExecution().getOwnerLease();
    }

    @Override
    public Mono<PlatformSubAgentRun> start(PlatformSubAgentCommand command) {
        return Mono.defer(() -> {
            PlatformSubAgentCommand safeCommand = Objects.requireNonNull(
                    command, "command must not be null");
            AgentKernelSnapshot snapshot = snapshots.build(safeCommand.kernelSpec());
            StartChildAgentRunCommand admission = new StartChildAgentRunCommand(
                    UUID.randomUUID().toString().replace("-", ""),
                    safeCommand.parentRunId(),
                    safeCommand.parentToolCallId(),
                    safeCommand.parentOwnerInstanceId(),
                    safeCommand.parentOwnerEpoch(),
                    safeCommand.agentName(),
                    safeCommand.kernelSpec().agentDefinitionStableKey(),
                    snapshot,
                    instanceIdentity.value(),
                    ownerLease,
                    safeCommand.deadline(),
                    inputHistory.userContent(safeCommand.messages()),
                    null);
            return coordinator.startChild(admission)
                    .flatMap(result -> projections
                            .projectCommitted(safeCommand.parentRunId())
                            .then(startIfCreated(safeCommand, snapshot, result)));
        });
    }

    @Override
    public Mono<PlatformSubAgentRun> awaitCompletion(
            PlatformSubAgentRun childRun) {
        return Mono.defer(() -> {
            PlatformSubAgentRun child = Objects.requireNonNull(
                    childRun, "childRun must not be null");
            return replay.replayThenLive(child.childRunId(), 0)
                    .takeUntil(event -> TERMINAL_OUTPUT_TYPES.contains(
                            event.outputType()))
                    .collectList()
                    .map(events -> completed(child, events));
        });
    }

    private PlatformSubAgentRun completed(
            PlatformSubAgentRun child,
            List<CommittedAgentEvent> events) {
        CommittedAgentEvent terminal = events.stream()
                .filter(event -> TERMINAL_OUTPUT_TYPES.contains(event.outputType()))
                .reduce((ignored, latest) -> latest)
                .orElseThrow(() -> new IllegalStateException(
                        "Child Agent replay completed without a terminal event: "
                                + child.childRunId()));
        AgentRunStatus status = switch (terminal.outputType()) {
            case "DONE" -> AgentRunStatus.COMPLETED;
            case "ERROR" -> AgentRunStatus.FAILED;
            case "CANCELLED" -> AgentRunStatus.CANCELLED;
            default -> throw new IllegalStateException(
                    "Unsupported child Agent terminal output: "
                            + terminal.outputType());
        };
        StringBuilder content = new StringBuilder();
        for (CommittedAgentEvent event : events) {
            if (!"CONTENT".equals(event.outputType())) {
                continue;
            }
            String delta = firstText(event.projection(), "delta", "content");
            if (delta != null) {
                content.append(delta);
            }
        }
        String error = status == AgentRunStatus.COMPLETED
                ? null
                : firstText(terminal.projection(), "error", "message");
        if (status == AgentRunStatus.CANCELLED && error == null) {
            error = "Child Agent run was cancelled";
        } else if (status == AgentRunStatus.FAILED && error == null) {
            error = "Child Agent run failed";
        }
        return new PlatformSubAgentRun(
                child.childRunId(),
                child.parentRunId(),
                child.parentToolCallId(),
                child.agentName(),
                status,
                content.isEmpty() ? null : content.toString(),
                error);
    }

    private String firstText(JsonNode payload, String... fields) {
        if (payload == null || !payload.isObject()) {
            return null;
        }
        for (String field : fields) {
            JsonNode value = payload.get(field);
            if (value != null && value.isTextual() && !value.textValue().isBlank()) {
                return value.textValue();
            }
        }
        return null;
    }

    private Mono<PlatformSubAgentRun> startIfCreated(
            PlatformSubAgentCommand command,
            AgentKernelSnapshot snapshot,
            ChildRunAdmission admission) {
        StartedAgentRun started = admission.run();
        PlatformSubAgentRun response = new PlatformSubAgentRun(
                started.runId(),
                command.parentRunId(),
                command.parentToolCallId(),
                command.agentName(),
                admission.status());
        if (!admission.created()) {
            return Mono.just(response);
        }
        return runtimeContexts.forChild(
                        started,
                        command.kernelSpec().agentDefinitionStableKey(),
                        command.projectContext(),
                        new ParentAgentRunContext(
                                command.parentRunId(),
                                command.parentOwnerInstanceId(),
                                command.parentOwnerEpoch(),
                                command.parentToolCallId(),
                                command.agentName()),
                        command.toolExecutionMode())
                .flatMap(runtime -> supervisor.start(new StartAgentExecutionCommand(
                        started,
                        command.messages(),
                        snapshot,
                        command.kernelSpec(),
                        runtime)))
                .thenReturn(response);
    }

    @Override
    public Mono<Void> cancelChildren(String parentRunId) {
        if (parentRunId == null || parentRunId.isBlank()) {
            return Mono.error(new IllegalArgumentException("parentRunId must not be blank"));
        }
        return cancellations.cancelChildren(parentRunId);
    }
}
