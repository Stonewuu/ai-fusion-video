package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.service.ai.agentscope.context.ParentAgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentCommand;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRun;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRunPort;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.*;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.Duration;
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
                    .filter(event -> !event.envelope().payload().path("_platformMirroredChildEvent").asBoolean(false))
                    .takeUntil(event -> TERMINAL_OUTPUT_TYPES.contains(
                            event.outputType()))
                    .collect(ChildOutput::new, ChildOutput::accept)
                    .map(output -> completed(child, output));
        });
    }

    private PlatformSubAgentRun completed(
            PlatformSubAgentRun child,
            ChildOutput output) {
        CommittedAgentEvent terminal = output.terminal;
        if (terminal == null) {
            throw new IllegalStateException(
                    "Child Agent replay completed without a terminal event: " + child.childRunId());
        }
        AgentRunStatus status = switch (terminal.outputType()) {
            case "DONE" -> AgentRunStatus.COMPLETED;
            case "ERROR" -> AgentRunStatus.FAILED;
            case "CANCELLED" -> AgentRunStatus.CANCELLED;
            default -> throw new IllegalStateException(
                    "Unsupported child Agent terminal output: "
                            + terminal.outputType());
        };
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
                output.content.isEmpty() ? null : output.content.toString(),
                error);
    }

    /**
     * 长任务只累计返回正文，不在父工具中复制全部思考、工具和事件历史。
     */
    private static final class ChildOutput {
        private final StringBuilder content = new StringBuilder();
        private CommittedAgentEvent terminal;

        private void accept(CommittedAgentEvent event) {
            if (TERMINAL_OUTPUT_TYPES.contains(event.outputType())) {
                terminal = event;
            } else if ("CONTENT".equals(event.outputType())) {
                JsonNode payload = event.projection();
                JsonNode value = payload.has("delta") ? payload.get("delta") : payload.get("content");
                if (value != null && value.isTextual()) {
                    content.append(value.textValue());
                }
            }
        }
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
                                command.agentName(),
                                "true".equals(command.kernelSpec().promptVariables().get("canvasNodeExecution"))),
                        command.toolExecutionMode())
                .flatMap(runtime -> supervisor.start(new StartAgentExecutionCommand(
                        started,
                        command.messages(),
                        snapshot,
                        command.kernelSpec(),
                        runtime)))
                .onErrorResume(OwnedExecutionRegistry.ExecutionAlreadyOwnedException.class,
                        alreadyStarted -> Mono.empty())
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
