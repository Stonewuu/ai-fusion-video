package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.agentscope.context.AgentConversationContext;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextRequest;
import com.stonewu.fusion.service.ai.agentscope.context.AuthenticatedUserContext;
import com.stonewu.fusion.service.ai.agentscope.context.CancellationContext;
import com.stonewu.fusion.service.ai.agentscope.context.PipelineRequestContext;
import com.stonewu.fusion.service.ai.agentscope.context.ProjectContext;
import com.stonewu.fusion.service.ai.agentscope.context.ToolExecutionContext;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.ResumedAgentRun;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.time.ZoneOffset;
import java.util.Objects;

@Component
public final class AgentExecutionRuntimeContextRequests {

    private final AgentRunRepository runRepository;
    private final AgentRuntimeSchedulers schedulers;

    public AgentExecutionRuntimeContextRequests(
            AgentRunRepository runRepository,
            AgentRuntimeSchedulers schedulers) {
        this.runRepository = Objects.requireNonNull(
                runRepository, "runRepository must not be null");
        this.schedulers = Objects.requireNonNull(schedulers, "schedulers must not be null");
    }

    public Mono<AgentScopeRuntimeContextRequest> forChild(
            StartedAgentRun started,
            String agentDefinitionStableKey,
            ProjectContext projectContext) {
        Objects.requireNonNull(started, "started must not be null");
        return load(started.runId()).map(run -> create(
                run,
                agentDefinitionStableKey,
                started.ownerInstanceId(),
                started.ownerEpoch(),
                started.deadline(),
                projectContext));
    }

    public Mono<AgentScopeRuntimeContextRequest> forResume(
            ResumedAgentRun resumed,
            String agentDefinitionStableKey) {
        Objects.requireNonNull(resumed, "resumed must not be null");
        return load(resumed.runId()).map(run -> create(
                run,
                agentDefinitionStableKey,
                resumed.newOwnerInstanceId(),
                resumed.newOwnerEpoch(),
                resumed.deadline(),
                run.getProjectId() != null ? new ProjectContext(run.getProjectId()) : null));
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
            ProjectContext requestedProject) {
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
        long userId = persisted.getUserId();
        return new AgentScopeRuntimeContextRequest(
                new AuthenticatedUserContext(userId),
                new AgentConversationContext(
                        persisted.getConversationId(),
                        agentDefinitionStableKey,
                        persisted.getAgentStateSessionId()),
                new AgentRunContext(
                        persisted.getRunId(), ownerInstanceId, ownerEpoch, deadline),
                project,
                new PipelineRequestContext(
                        persisted.getRunId(), PipelineRequestContext.Kind.PIPELINE),
                new ToolExecutionContext(userId, 1, userId),
                CancellationContext.noop());
    }
}
