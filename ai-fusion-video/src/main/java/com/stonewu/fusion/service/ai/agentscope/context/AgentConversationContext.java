package com.stonewu.fusion.service.ai.agentscope.context;

public record AgentConversationContext(String conversationId, String agentDefinitionStableKey) {

    public AgentConversationContext {
        conversationId = ContextValues.requireSessionComponent(conversationId, "conversationId");
        agentDefinitionStableKey = ContextValues.requireSessionComponent(
                agentDefinitionStableKey, "agentDefinitionStableKey");
    }
}
