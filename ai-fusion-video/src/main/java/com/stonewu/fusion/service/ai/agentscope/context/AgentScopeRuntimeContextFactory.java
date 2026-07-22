package com.stonewu.fusion.service.ai.agentscope.context;

import io.agentscope.core.agent.RuntimeContext;
import org.springframework.stereotype.Component;

import java.util.Objects;

@Component
public final class AgentScopeRuntimeContextFactory {

    private static final String SESSION_PREFIX = "afv:v2:";

    public RuntimeContext create(AgentScopeRuntimeContextRequest request) {
        AgentScopeRuntimeContextRequest safeRequest = Objects.requireNonNull(
                request, "request must not be null");

        RuntimeContext.Builder builder = RuntimeContext.builder()
                .userId(String.valueOf(safeRequest.authenticatedUser().userId()))
                .sessionId(sessionId(safeRequest.conversation()))
                .put(AuthenticatedUserContext.class, safeRequest.authenticatedUser())
                .put(AgentConversationContext.class, safeRequest.conversation())
                .put(AgentRunContext.class, safeRequest.run())
                .put(PipelineRequestContext.class, safeRequest.pipelineRequest())
                .put(CancellationContext.class, safeRequest.cancellation());

        if (safeRequest.project() != null) {
            builder.put(ProjectContext.class, safeRequest.project());
        }
        if (safeRequest.parentRun() != null) {
            builder.put(ParentAgentRunContext.class, safeRequest.parentRun());
        }
        if (safeRequest.toolExecution() != null) {
            builder.put(ToolExecutionContext.class, safeRequest.toolExecution());
        }
        return builder.build();
    }

    private String sessionId(AgentConversationContext conversation) {
        if (conversation.agentStateSessionId() != null) {
            return conversation.agentStateSessionId();
        }
        return SESSION_PREFIX + conversation.conversationId() + ':' + conversation.agentDefinitionStableKey();
    }
}
