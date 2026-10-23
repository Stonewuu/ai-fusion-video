package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.ai.agentscope.context.ParentAgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.ProjectContext;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.message.AgentScopeMessageMapper;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.run.*;
import com.stonewu.fusion.service.ai.run.kernel.RunConfigUnavailableException;
import com.stonewu.fusion.service.ai.run.model.StartAgentExecutionCommand;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

@Component
@RequiredArgsConstructor
@Slf4j
public class CanvasAgentChildDispatch {
    private final CanvasAgentChildQueue queue;
    private final AgentRuntimeInstanceIdentity identity;
    private final ObjectProvider<AgentExecutionFactory> executions;
    private final ObjectProvider<RunExecutionSupervisor> supervisor;
    private final AgentExecutionRuntimeContextRequests contexts;
    private final CanvasDocumentRepository canvases;
    private final CanvasAccessPolicy access;
    private final AgentScopeMessageMapper messages;
    private final CanvasAgentNodeDispatch nodeDispatch;
    private final Set<String> active = ConcurrentHashMap.newKeySet();
    private final Semaphore slots = new Semaphore(4);
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();

    @Scheduled(fixedDelay = 1000)
    public void tick() {
        for (String id : queue.pending(identity.value())) {
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
                    log.warn("专业子任务启动暂未完成 [{}]: {}", id, failure.getClass().getSimpleName());
                } finally {
                    active.remove(id);
                    slots.release();
                }
            });
        }
    }

    public void dispatch(String id) {
        var dispatch = queue.claim(id, identity.value());
        if (dispatch == null) {
            return;
        }
        try {
            access.requireExecute(canvases.require(dispatch.canvasId(), false), dispatch.actor());
            var child = dispatch.child();
            var parent = dispatch.parent();
            var snapshot = dispatch.started().kernelSnapshot();
            var permission = ToolExecutionMode.parse(
                    snapshot.payload().promptVariables().get(AgentKernelSpecFactory.TOOL_EXECUTION_MODE_VARIABLE));
            var parentContext = new ParentAgentRunContext(parent.getRunId(), parent.getOwnerInstanceId(),
                    parent.getOwnerEpoch(),
                    child.getParentToolCallId(), child.getAgentName(), true);
            executions.getObject().resolve(snapshot)
                    .flatMap(spec -> contexts.forChild(dispatch.started(), spec.agentDefinitionStableKey(),
                                    child.getProjectId() == null ? null : new ProjectContext(child.getProjectId()),
                                    parentContext, permission)
                            .flatMap(runtime -> supervisor.getObject()
                                    .start(new StartAgentExecutionCommand(dispatch.started(),
                                            messages.toUserMessages(dispatch.input()),
                                            snapshot, spec, runtime))))
                    .onErrorResume(OwnedExecutionRegistry.ExecutionAlreadyOwnedException.class,
                            alreadyStarted -> Mono.empty())
                    .block(Duration.ofSeconds(30));
        } catch (BusinessException invalid) {
            if (!Set.of(400, 403, 404, 409, 410, 422).contains(invalid.getCode())) {
                throw invalid;
            }
            nodeDispatch.failStarted(dispatch.started(), dispatch.actor().userId(), invalid.getMessage());
        } catch (RunConfigUnavailableException invalid) {
            nodeDispatch.failStarted(dispatch.started(), dispatch.actor().userId(), invalid.getMessage());
        }
    }

    @PreDestroy
    public void close() {
        workers.shutdownNow();
    }
}
