package com.stonewu.fusion.service.ai.agentscope.context;

public record AuthenticatedUserContext(Long userId) {

    public AuthenticatedUserContext {
        userId = ContextValues.requirePositive(userId, "userId");
    }
}
