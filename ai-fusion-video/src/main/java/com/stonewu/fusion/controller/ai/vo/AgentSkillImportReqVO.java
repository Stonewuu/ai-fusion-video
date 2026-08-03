package com.stonewu.fusion.controller.ai.vo;

import com.stonewu.fusion.service.ai.agentscope.skill.AgentSkillImportService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;

import java.util.List;

public record AgentSkillImportReqVO(
        @NotEmpty List<@Valid Selection> selections) {

    public record Selection(
            @NotBlank String rootPath,
            @NotBlank String displayName,
            @NotNull AgentSkillImportService.ImportAction action) {
    }
}
