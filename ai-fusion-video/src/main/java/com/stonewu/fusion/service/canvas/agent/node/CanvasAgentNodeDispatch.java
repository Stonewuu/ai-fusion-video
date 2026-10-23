package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentRuntimeErrorCode;
import com.stonewu.fusion.enums.ai.AgentTerminalOutputType;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.ai.agentscope.context.ProjectContext;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import com.stonewu.fusion.service.ai.run.*;
import com.stonewu.fusion.service.ai.run.kernel.RunConfigUnavailableException;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.StartAgentExecutionCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

@Slf4j
@Component
@RequiredArgsConstructor
public class CanvasAgentNodeDispatch {
    private final CanvasAgentNodeRepository jobs;
    private final CanvasRunRepository runs;
    private final CanvasDocumentRepository canvases;
    private final CanvasAccessPolicy access;
    private final CanvasRunSources sources;
    private final AgentRuntimeInstanceIdentity identity;
    private final ObjectProvider<AgentExecutionFactory> executions;
    private final ObjectProvider<RunExecutionSupervisor> supervisor;
    private final ObjectProvider<RunTerminalCoordinator> terminals;
    private final ObjectProvider<AgentMessageProjectionService> projections;
    private final AgentExecutionRuntimeContextRequests contexts;
    private final CanvasAgentNodeInputs inputs;
    private final CanvasJson json;
    private final Set<String> active = ConcurrentHashMap.newKeySet();
    private final Semaphore slots = new Semaphore(4);
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();

    @Scheduled(fixedDelay = 1000)
    public void tick() {
        for (String id : jobs.pending(identity.value())) {
            if (!active.add(id)) {
                continue;
            }
            if (!slots.tryAcquire()) {
                active.remove(id);
                return;
            }
            workers.submit(() -> {
                try {
                    dispatch(id);
                } catch (RuntimeException failure) {
                    log.warn("专业画布智能体启动未完成 [{}]: {}", id, failure.getMessage());
                } finally {
                    active.remove(id);
                    slots.release();
                }
            });
        }
    }

    public void dispatch(String id) {
        var dispatch = jobs.claim(id, identity.value());
        if (dispatch == null) {
            return;
        }
        var run = runs.run(dispatch.job().canvasRunId(), false);
        try {
            start(dispatch, run);
        } catch (BusinessException invalid) {
            if (!Set.of(400, 403, 404, 409, 410, 422).contains(invalid.getCode())) {
                throw invalid;
            }
            fail(dispatch, run.actor().userId(), invalid.getMessage());
        } catch (RunConfigUnavailableException invalid) {
            fail(dispatch, run.actor().userId(), invalid.getMessage());
        }
    }

    private void start(CanvasAgentNodeRepository.Dispatch dispatch, Run run) {
        var node = runs.node(dispatch.job().nodeRunId(), false);
        access.requireExecute(canvases.require(run.canvasId(), false), run.actor());
        if (run.cancelRequested()) {
            throw new BusinessException(409, "画布运行已经停止");
        }
        node.inputs().all().forEach(value -> sources.authorize(value, run.actor(), true));
        var started = dispatch.started();
        var snapshot = started.kernelSnapshot();
        var mode = ToolExecutionMode.parse(
                snapshot.payload().promptVariables().get(AgentKernelSpecFactory.TOOL_EXECUTION_MODE_VARIABLE));
        executions.getObject().resolve(snapshot)
                .flatMap(spec -> contexts.forRoot(started, spec.agentDefinitionStableKey(),
                                run.plan().graph().projectId() == null ? null : new ProjectContext(
                                        run.plan().graph().projectId()), mode)
                        .flatMap(runtime -> supervisor.getObject().start(new StartAgentExecutionCommand(started,
                                inputs.messages(run, node, dispatch.job().input(), spec.model()), snapshot, spec,
                                runtime))))
                .onErrorResume(OwnedExecutionRegistry.ExecutionAlreadyOwnedException.class,
                        alreadyRunning -> Mono.empty())
                .block(Duration.ofSeconds(30));
        jobs.dispatched(dispatch.job().nodeRunId(), started.ownerEpoch());
    }

    private void fail(CanvasAgentNodeRepository.Dispatch dispatch, long userId, String message) {
        failStarted(dispatch.started(), userId, message);
        jobs.dispatched(dispatch.job().nodeRunId(), dispatch.started().ownerEpoch());
    }

    public void failStarted(StartedAgentRun run, long userId, String message) {
        var payload = json.object(
                Map.of("outputType", "ERROR", "errorCode", "RUN_CONFIG_UNAVAILABLE", "error", message, "finished",
                        true));
        var event = new AgentEventEnvelope("canvas-node-start-" + UUID.randomUUID(), "RUN_TERMINAL", "main", null, null,
                null,
                null, null, "ERROR", payload, Instant.now());
        var request = new RunTerminalRequest(run.runId(),
                new StateStoreSlot(String.valueOf(userId), run.agentStateSessionId()),
                Set.of(AgentRunStatus.RUNNING), AgentRunStatus.FAILED, AgentTerminalOutputType.ERROR,
                AgentRuntimeErrorCode.RUN_CONFIG_UNAVAILABLE, message, event);
        terminals.getObject().terminateOwned(request, run.ownerInstanceId(), run.ownerEpoch())
                .flatMap(result -> result.isPresent() ? projections.getObject()
                        .projectThrough(result.get().runId(), result.get().sequence()) : Mono.empty())
                .block(Duration.ofSeconds(30));
    }

    @PreDestroy
    public void close() {
        workers.shutdownNow();
    }
}
