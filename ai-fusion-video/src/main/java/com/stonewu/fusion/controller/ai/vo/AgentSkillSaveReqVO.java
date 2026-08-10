package com.stonewu.fusion.controller.ai.vo;

import jakarta.validation.constraints.NotBlank;

public record AgentSkillSaveReqVO(
        String originalName,
        @NotBlank String name,
        @NotBlank String displayName,
        @NotBlank String description,
        @NotBlank String content) {
}
