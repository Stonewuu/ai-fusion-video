package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentRuntimeErrorCode;
import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreFailure;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreFailureGuard;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.SystemTerminalActor;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/** Reactive terminal adapter with fail-closed StateStore completion checks. */
@Service
@RequiredArgsConstructor
public class MySqlRunTerminalCoordinator implements RunTerminalCoordinator {

    private static final String STATE_STORE_FAILURE_MESSAGE =
            "Agent state persistence failed before completion";

    private final AgentEventRepository repository;
    private final AgentRunMapper runMapper;
    private final StateStoreFailureGuard stateStoreFailureGuard;
    private final AgentRuntimeSchedulers schedulers;
    private final AgentRunRedisSignalService signals;
    private AgentRuntimeMetrics metrics = AgentRuntimeMetrics.noop();

    @Autowired
    void setMetrics(AgentRuntimeMetrics metrics) {
        this.metrics = Objects.requireNonNull(
                metrics, "metrics must not be null");
    }

    @Override
    public Mono<Optional<CommittedAgentEvent>> terminateOwned(
            RunTerminalRequest request,
            String ownerInstanceId,
            long ownerEpoch) {
        return Mono.fromCallable(() -> repository.terminateOwnedTx(
                        failClosedCompletion(request), ownerInstanceId, ownerEpoch))
                .subscribeOn(schedulers.journal())
                .flatMap(this::afterTerminalCommit);
    }

    @Override
    public Mono<Optional<CommittedAgentEvent>> terminateSystem(
            RunTerminalRequest request,
            SystemTerminalActor actor) {
        return Mono.fromCallable(() -> repository.terminateSystemTx(request, actor))
                .subscribeOn(schedulers.journal())
                .flatMap(this::afterTerminalCommit);
    }

    private Mono<Optional<CommittedAgentEvent>> afterTerminalCommit(
            Optional<CommittedAgentEvent> committed) {
        recordTerminal(committed);
        return committed.filter(CommittedAgentEvent::publishRequired)
                .map(event -> signals.publishWakeup(event.runId(), event.sequence())
                        .onErrorResume(failure -> Mono.empty())
                        .thenReturn(committed))
                .orElseGet(() -> Mono.just(committed));
    }

    private void recordTerminal(Optional<CommittedAgentEvent> committed) {
        committed.ifPresent(event -> metrics.terminal(switch (event.outputType()) {
            case "DONE" -> AgentRunStatus.COMPLETED;
            case "ERROR" -> AgentRunStatus.FAILED;
            case "CANCELLED" -> AgentRunStatus.CANCELLED;
            default -> throw new IllegalStateException(
                    "Unknown Agent terminal output type: " + event.outputType());
        }));
    }

    private RunTerminalRequest failClosedCompletion(RunTerminalRequest request) {
        if (request == null) {
            throw new IllegalArgumentException("request must not be null");
        }
        AgentRun run = runMapper.selectOne(new LambdaQueryWrapper<AgentRun>()
                .select(AgentRun::getUserId, AgentRun::getAgentStateSessionId)
                .eq(AgentRun::getRunId, request.runId()));
        if (run != null) {
            StateStoreSlot authoritativeSlot = new StateStoreSlot(
                    String.valueOf(run.getUserId()), run.getAgentStateSessionId());
            if (!authoritativeSlot.equals(request.stateStoreSlot())) {
                throw new IllegalArgumentException(
                        "Run terminal StateStore slot does not match persisted run identity");
            }
        }
        if (request.terminalStatus() != AgentRunStatus.COMPLETED) {
            return request;
        }
        try {
            stateStoreFailureGuard.throwIfFailed(request.stateStoreSlot());
            return request;
        } catch (StateStoreFailure failure) {
            return request.asFailure(
                    AgentRuntimeErrorCode.STATE_STORE_FAILED,
                    STATE_STORE_FAILURE_MESSAGE,
                    stateStoreFailureEnvelope(request.terminalEnvelope()));
        }
    }

    private AgentEventEnvelope stateStoreFailureEnvelope(AgentEventEnvelope original) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", "ERROR")
                .put("errorCode", AgentRuntimeErrorCode.STATE_STORE_FAILED.getCode())
                .put("error", STATE_STORE_FAILURE_MESSAGE)
                .put("finished", true);
        return new AgentEventEnvelope(
                "state-store-failure-" + UUID.randomUUID().toString().replace("-", ""),
                "STATE_STORE_FAILED",
                original.source(),
                original.replyId(),
                original.blockId(),
                original.toolCallId(),
                original.parentToolCallId(),
                original.agentName(),
                "ERROR",
                payload,
                Instant.now());
    }
}
