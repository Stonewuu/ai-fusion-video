package com.stonewu.fusion.service.ai.configuration;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Size;

import java.util.List;

/**
 * null 继承内置值；空列表明确停用该组工具或 Skill。
 */
public record AgentSettingsOverride(
        @Size(min = 1, max = 100) String name,
        @Size(min = 1, max = 131072) String systemPrompt,
        @Size(max = 65536) String instructionTemplate,
        @Size(max = 32768) String defaultUserMessage,
        Boolean toolsEnabled,
        @Size(max = 256) List<@Size(min = 1, max = 150) String> toolNames,
        @Size(max = 64) List<@Size(min = 1, max = 150) String> subAgentToolNames,
        @Size(max = 256) List<@Size(min = 1, max = 200) String> mcpToolNames,
        @Size(max = 8) List<@Size(min = 1, max = 100) String> defaultSkills,
        @Min(0) Integer maxIters) {
    public static AgentSettingsOverride inherit() {
        return new AgentSettingsOverride(null, null, null, null, null, null, null, null, null, null);
    }
}
