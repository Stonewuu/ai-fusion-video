package com.stonewu.fusion.controller.ai.vo;

import com.stonewu.fusion.entity.ai.AgentWorkspaceMigration;

public record AgentWorkspaceConfigRespVO(
        String backendType,
        Long storageConfigId,
        String localPath,
        String migrationStatus,
        Long activeMigrationId,
        long entryCount,
        long contentBytes,
        AgentWorkspaceMigration latestMigration) {
}
