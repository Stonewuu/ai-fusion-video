package com.stonewu.fusion.service.ai.agentscope.workspace;

public record AgentWorkspaceLocation(
        String backendType,
        Long storageConfigId,
        String localPath) {

    public AgentWorkspaceLocation {
        backendType = AgentWorkspaceBackend.requireSupported(backendType);
        localPath = localPath == null || localPath.isBlank() ? null : localPath.trim();
    }
}
