package com.stonewu.fusion.service.ai.agentscope.context;

public record ToolExecutionContext(Long userId, Integer ownerType, Long ownerId) {

    public ToolExecutionContext {
        userId = ContextValues.requirePositive(userId, "userId");
        ownerType = ContextValues.requirePositive(ownerType, "ownerType");
        ownerId = ContextValues.requirePositive(ownerId, "ownerId");
    }
}
