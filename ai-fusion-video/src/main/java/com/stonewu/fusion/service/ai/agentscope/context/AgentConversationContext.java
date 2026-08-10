package com.stonewu.fusion.service.ai.agentscope.context;

public record AgentConversationContext(
        String conversationId,
        String agentDefinitionStableKey,
        String agentStateSessionId) {

    public AgentConversationContext(String conversationId, String agentDefinitionStableKey) {
        this(conversationId, agentDefinitionStableKey, null);
    }

    public AgentConversationContext {
        conversationId = ContextValues.requireSessionComponent(conversationId, "conversationId");
        agentDefinitionStableKey = ContextValues.requireSessionComponent(
                agentDefinitionStableKey, "agentDefinitionStableKey");
        if (agentStateSessionId != null) {
            agentStateSessionId = ContextValues.requireText(
                    agentStateSessionId, "agentStateSessionId");
        }
    }
}
