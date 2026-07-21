package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentCommand;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRun;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRunPort;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.ChildRunAdmission;
import com.stonewu.fusion.service.ai.run.model.ExecutionStopReason;
import com.stonewu.fusion.service.ai.run.model.StartAgentExecutionCommand;
import com.stonewu.fusion.service.ai.run.model.StartChildAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.UUID;

@Service
public final class PlatformSubAgentRunService implements PlatformSubAgentRunPort {

    private final AgentRunCoordinator coordinator;
    private final AgentKernelSnapshotBuilder snapshots;
    private final AgentInputHistoryMapper inputHistory;
    private final AgentExecutionRuntimeContextRequests runtimeContexts;
    private final RunExecutionSupervisor supervisor;
    private final AgentRunRepository runRepository;
    private final AgentRunRedisSignalService signals;
    private final AgentRuntimeSchedulers schedulers;
    private final TransactionTemplate transactions;
    private final AgentRuntimeInstanceIdentity instanceIdentity;
    private final Duration ownerLease;

    public PlatformSubAgentRunService(
            AgentRunCoordinator coordinator,
            AgentKernelSnapshotBuilder snapshots,
            AgentInputHistoryMapper inputHistory,
            AgentExecutionRuntimeContextRequests runtimeContexts,
            RunExecutionSupervisor supervisor,
            AgentRunRepository runRepository,
            AgentRunRedisSignalService signals,
            AgentRuntimeSchedulers schedulers,
            TransactionTemplate transactions,
            AgentRuntimeInstanceIdentity instanceIdentity,
            AgentScopeV2Properties properties) {
        this.coordinator = Objects.requireNonNull(coordinator, "coordinator must not be null");
        this.snapshots = Objects.requireNonNull(snapshots, "snapshots must not be null");
        this.inputHistory = Objects.requireNonNull(inputHistory, "inputHistory must not be null");
        this.runtimeContexts = Objects.requireNonNull(runtimeContexts, "runtimeContexts must not be null");
        this.supervisor = Objects.requireNonNull(supervisor, "supervisor must not be null");
        this.runRepository = Objects.requireNonNull(runRepository, "runRepository must not be null");
        this.signals = Objects.requireNonNull(signals, "signals must not be null");
        this.schedulers = Objects.requireNonNull(schedulers, "schedulers must not be null");
        this.transactions = Objects.requireNonNull(transactions, "transactions must not be null");
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
                    .flatMap(result -> startIfCreated(safeCommand, snapshot, result));
        });
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
                        command.projectContext())
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
        return Mono.fromCallable(() -> Objects.requireNonNull(
                        transactions.execute(ignored ->
                                runRepository.requestCancelActiveDescendants(parentRunId)),
                        "child cancellation transaction returned null"))
                .subscribeOn(schedulers.journal())
                .flatMapMany(Flux::fromIterable)
                .concatMap(this::notifyAndInterrupt)
                .then();
    }

    private Mono<Void> notifyAndInterrupt(AgentRun child) {
        if (child.getOwnerInstanceId() == null || child.getOwnerEpoch() == null) {
            return signals.publishCancel(child.getRunId());
        }
        return signals.publishCancel(child.getRunId())
                .onErrorResume(ignored -> Mono.empty())
                .then(supervisor.interruptOwned(
                        child.getRunId(),
                        child.getOwnerInstanceId(),
                        child.getOwnerEpoch(),
                        ExecutionStopReason.CANCEL_REQUESTED))
                .then();
    }
}
