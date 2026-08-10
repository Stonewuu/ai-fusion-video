package com.stonewu.fusion.service.ai.agentscope.context;

public record ProjectContext(Long projectId) {

    public ProjectContext {
        projectId = ContextValues.requirePositive(projectId, "projectId");
    }
}
