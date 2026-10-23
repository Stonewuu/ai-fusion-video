package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.agentscope.context.*;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.ResumedAgentRun;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScopeService;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.time.ZoneOffset;
import java.util.Objects;

@Component
public final class AgentExecutionRuntimeContextRequests {

    private final AgentRunRepository runRepository;
    private final AgentRuntimeSchedulers schedulers;
    private final CanvasAgentScopeService canvasScopes;

    public AgentExecutionRuntimeContextRequests(
            AgentRunRepository runRepository,
            AgentRuntimeSchedulers schedulers,
            CanvasAgentScopeService canvasScopes) {
        this.runRepository = Objects.requireNonNull(
                runRepository, "runRepository must not be null");
        this.schedulers = Objects.requireNonNull(schedulers, "schedulers must not be null");
        this.canvasScopes = Objects.requireNonNull(canvasScopes, "canvasScopes must not be null");
    }

    public Mono<AgentScopeRuntimeContextRequest> forChild(
            StartedAgentRun started,
            String agentDefinitionStableKey,
            ProjectContext projectContext,
            ParentAgentRunContext parentRun,
            ToolExecutionMode toolExecutionMode) {
        Objects.requireNonNull(started, "started must not be null");
        Objects.requireNonNull(parentRun, "parentRun must not be null");
        Objects.requireNonNull(toolExecutionMode, "toolExecutionMode must not be null");
        return load(started.runId()).map(run -> create(
                run,
                agentDefinitionStableKey,
                started.ownerInstanceId(),
                started.ownerEpoch(),
                started.deadline(),
                projectContext,
                parentRun,
                toolExecutionMode,
                PipelineRequestContext.Kind.PIPELINE));
    }

    public Mono<AgentScopeRuntimeContextRequest> forRoot(
            StartedAgentRun started,
            String agentDefinitionStableKey,
            ProjectContext projectContext,
            ToolExecutionMode toolExecutionMode) {
        Objects.requireNonNull(started, "started must not be null");
        Objects.requireNonNull(toolExecutionMode, "toolExecutionMode must not be null");
        return load(started.runId()).map(run -> {
            if (run.getParentRunId() != null || run.getParentToolCallId() != null) {
                throw new IllegalArgumentException(
                        "Root RuntimeContext cannot be created for a child run");
            }
            return create(
                    run,
                    agentDefinitionStableKey,
                    started.ownerInstanceId(),
                    started.ownerEpoch(),
                    started.deadline(),
                    projectContext,
                    null,
                    toolExecutionMode,
                    PipelineRequestContext.Kind.PIPELINE);
        });
    }

    public Mono<AgentScopeRuntimeContextRequest> forResume(
            ResumedAgentRun resumed,
            String agentDefinitionStableKey,
            ToolExecutionMode toolExecutionMode) {
        Objects.requireNonNull(resumed, "resumed must not be null");
        Objects.requireNonNull(toolExecutionMode, "toolExecutionMode must not be null");
        return load(resumed.runId()).map(run -> create(
                run,
                agentDefinitionStableKey,
                resumed.newOwnerInstanceId(),
                resumed.newOwnerEpoch(),
                resumed.deadline(),
                run.getProjectId() != null ? new ProjectContext(run.getProjectId()) : null,
                parentForResume(run),
                toolExecutionMode,
                PipelineRequestContext.Kind.PIPELINE));
    }

    private Mono<AgentRun> load(String runId) {
        return Mono.fromCallable(() -> {
                    AgentRun run = runRepository.findRun(runId);
                    if (run == null) {
                        throw new IllegalStateException("Agent run does not exist: " + runId);
                    }
                    return run;
                })
                .subscribeOn(schedulers.journal());
    }

    private AgentScopeRuntimeContextRequest create(
            AgentRun persisted,
            String agentDefinitionStableKey,
            String ownerInstanceId,
            long ownerEpoch,
            Instant deadline,
            ProjectContext requestedProject,
            ParentAgentRunContext parentRun,
            ToolExecutionMode toolExecutionMode,
            PipelineRequestContext.Kind requestKind) {
        Objects.requireNonNull(toolExecutionMode, "toolExecutionMode must not be null");
        Objects.requireNonNull(requestKind, "requestKind must not be null");
        if (!Objects.equals(persisted.getAgentType(), agentDefinitionStableKey)) {
            throw new IllegalArgumentException(
                    "RuntimeContext agent definition does not match the persisted run");
        }
        if (!persisted.getDeadlineAt().toInstant(ZoneOffset.UTC).equals(deadline)) {
            throw new IllegalArgumentException(
                    "RuntimeContext deadline does not match the persisted run deadline");
        }
        ProjectContext project = persisted.getProjectId() != null
                ? new ProjectContext(persisted.getProjectId())
                : null;
        if (requestedProject != null && !requestedProject.equals(project)) {
            throw new IllegalArgumentException(
                    "RuntimeContext project does not match the persisted run project");
        }
        validateParent(persisted, parentRun);
        long userId = persisted.getUserId();
        var canvasScope = canvasScopes.decode(persisted.getCanvasScopeJson());
        // A resumed canvas node executes the same admitted workflow, including its automatic approval policy.
        var effectiveMode = canvasScope != null && canvasScope.nodeRunId() != null
                ? ToolExecutionMode.FULL_ACCESS : toolExecutionMode;
        return new AgentScopeRuntimeContextRequest(
                new AuthenticatedUserContext(userId),
                new AgentConversationContext(
                        persisted.getConversationId(),
                        agentDefinitionStableKey,
                        persisted.getAgentStateSessionId()),
                new AgentRunContext(
                        persisted.getRunId(),
                        ownerInstanceId,
                        ownerEpoch,
                        deadline,
                        persisted.getParentToolCallId(),
                        persisted.getAgentName()),
                parentRun,
                project,
                new PipelineRequestContext(
                        persisted.getRunId(), requestKind),
                new ToolExecutionContext(userId, persisted.getActorTeamId() == null ? 1 : 2,
                        persisted.getActorTeamId() == null ? userId : persisted.getActorTeamId(),
                        canvasScope),
                CancellationContext.noop(),
                new ToolPermissionContext(effectiveMode));
    }

    private ParentAgentRunContext parentForResume(AgentRun child) {
        if (child.getParentRunId() == null) {
            return null;
        }
        AgentRun parent = runRepository.findRun(child.getParentRunId());
        var canvasScope = canvasScopes.decode(child.getCanvasScopeJson());
        boolean independent = canvasScope != null && canvasScope.nodeRunId() != null;
        if (parent == null
                || (!independent && (parent.getOwnerInstanceId() == null || parent.getOwnerInstanceId().isBlank()))
                || parent.getOwnerEpoch() == null
                || parent.getOwnerEpoch() <= 0) {
            throw new IllegalStateException(
                    "Child Agent run has no active parent owner: "
                            + child.getRunId());
        }
        return new ParentAgentRunContext(
                parent.getRunId(),
                parent.getOwnerInstanceId(),
                parent.getOwnerEpoch(),
                child.getParentToolCallId(),
                child.getAgentName(), independent);
    }

    private void validateParent(
            AgentRun persisted, ParentAgentRunContext parentRun) {
        if (persisted.getParentRunId() == null) {
            if (parentRun != null) {
                throw new IllegalArgumentException(
                        "Root RuntimeContext must not carry a parent run");
            }
            return;
        }
        if (parentRun == null
                || !Objects.equals(persisted.getParentRunId(), parentRun.runId())
                || !Objects.equals(
                persisted.getParentToolCallId(), parentRun.toolCallId())
                || !Objects.equals(persisted.getAgentName(), parentRun.agentName())) {
            throw new IllegalArgumentException(
                    "Child RuntimeContext parent identity does not match the persisted run");
        }
    }
}
