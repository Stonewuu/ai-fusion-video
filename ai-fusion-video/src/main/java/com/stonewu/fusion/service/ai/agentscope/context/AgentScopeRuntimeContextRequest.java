package com.stonewu.fusion.service.ai.agentscope.context;

import java.util.Objects;

public record AgentScopeRuntimeContextRequest(
        AuthenticatedUserContext authenticatedUser,
        AgentConversationContext conversation,
        AgentRunContext run,
        ParentAgentRunContext parentRun,
        ProjectContext project,
        PipelineRequestContext pipelineRequest,
        ToolExecutionContext toolExecution,
        CancellationContext cancellation) {

    public AgentScopeRuntimeContextRequest {
        authenticatedUser = Objects.requireNonNull(authenticatedUser, "authenticatedUser must not be null");
        conversation = Objects.requireNonNull(conversation, "conversation must not be null");
        run = Objects.requireNonNull(run, "run must not be null");
        pipelineRequest = Objects.requireNonNull(pipelineRequest, "pipelineRequest must not be null");
        cancellation = Objects.requireNonNull(cancellation, "cancellation must not be null");
    }

    public AgentScopeRuntimeContextRequest(
            AuthenticatedUserContext authenticatedUser,
            AgentConversationContext conversation,
            AgentRunContext run,
            ProjectContext project,
            PipelineRequestContext pipelineRequest,
            ToolExecutionContext toolExecution,
            CancellationContext cancellation) {
        this(authenticatedUser, conversation, run, null, project,
                pipelineRequest, toolExecution, cancellation);
    }
}
