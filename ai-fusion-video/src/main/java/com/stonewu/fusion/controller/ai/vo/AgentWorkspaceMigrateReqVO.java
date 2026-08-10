package com.stonewu.fusion.controller.ai.vo;

import jakarta.validation.constraints.NotBlank;

public record AgentWorkspaceMigrateReqVO(
        @NotBlank String backendType,
        Long storageConfigId,
        String localPath) {
}
