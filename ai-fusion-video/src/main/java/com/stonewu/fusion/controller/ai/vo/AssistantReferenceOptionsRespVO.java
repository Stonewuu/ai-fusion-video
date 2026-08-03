package com.stonewu.fusion.controller.ai.vo;

import java.util.List;

public record AssistantReferenceOptionsRespVO(
        List<SkillOption> skills,
        List<McpToolOption> mcpTools) {

    public AssistantReferenceOptionsRespVO {
        skills = List.copyOf(skills);
        mcpTools = List.copyOf(mcpTools);
    }

    public record SkillOption(
            String id,
            String name,
            String displayName,
            String description,
            String source) {
    }

    public record McpToolOption(
            String serverName,
            String toolName,
            String description,
            boolean readOnly) {
    }
}
