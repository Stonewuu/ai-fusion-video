package com.stonewu.fusion.service.ai.configuration;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.config.ai.AiAgentDefinition;
import com.stonewu.fusion.config.ai.AiAgentRegistry;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 内置定义保持只读，按可信用户标识生成独立的生效配置。
 */
@Component
@RequiredArgsConstructor
public class AgentDefinitionConfiguration {
    public static final String SNAPSHOT_VARIABLE = "agentUserConfiguration";

    public record Values(String name, String systemPrompt, String instructionTemplate, String defaultUserMessage,
                         boolean toolsEnabled, List<String> toolNames, List<String> subAgentToolNames,
                         List<String> mcpToolNames, List<String> defaultSkills, int maxIters) {
    }

    public record Resolved(String type, long revision, AgentSettingsOverride overrides, Values defaults,
                           Values effective) {
    }

    private final AiAgentRegistry registry;
    private final AgentUserOverrideRepository repository;
    private final ObjectMapper mapper;
    private final AgentScopeV2Properties properties;

    public AiAgentDefinition builtin(String type) {
        var definition = registry.getByType(type);
        if (definition == null) {
            throw new BusinessException(404, "智能体不存在：" + type);
        }
        return definition;
    }

    public Resolved get(String type, Long userId) {
        var base = builtin(type);
        var stored = userId == null ? new AgentUserOverrideRepository.Stored(0, "{}") : repository.get(userId, type);
        AgentSettingsOverride overrides;
        try {
            overrides = mapper.readValue(stored.json(), AgentSettingsOverride.class);
        } catch (JsonProcessingException failure) {
            throw new IllegalStateException("用户智能体配置损坏", failure);
        }
        var defaults = new Values(base.getName(), base.getSystemPrompt(), text(base.getInstructionTemplate()),
                text(base.getDefaultUserMessage()),
                Integer.valueOf(1).equals(base.getEnableTools()), base.getToolNames(),
                base.getSubAgentTools() == null ? List.of() : base.getSubAgentTools().stream()
                        .map(AiAgentDefinition.SubAgentToolDef::getToolName).toList(),
                base.getMcpToolNames(), base.getDefaultSkills(),
                base.getMaxIters() == null ? properties.getExecution().getMaxIters() : base.getMaxIters());
        return new Resolved(type, stored.revision(), overrides, defaults, merge(defaults, overrides));
    }

    public Values merge(Values base, AgentSettingsOverride value) {
        return new Values(value.name() == null ? base.name() : value.name(),
                value.systemPrompt() == null ? base.systemPrompt() : value.systemPrompt(),
                value.instructionTemplate() == null ? base.instructionTemplate() : value.instructionTemplate(),
                value.defaultUserMessage() == null ? base.defaultUserMessage() : value.defaultUserMessage(),
                value.toolsEnabled() == null ? base.toolsEnabled() : value.toolsEnabled(),
                value.toolNames() == null ? base.toolNames() : List.copyOf(value.toolNames()),
                value.subAgentToolNames() == null ? base.subAgentToolNames() : List.copyOf(value.subAgentToolNames()),
                value.mcpToolNames() == null ? base.mcpToolNames() : List.copyOf(value.mcpToolNames()),
                value.defaultSkills() == null ? base.defaultSkills() : List.copyOf(value.defaultSkills()),
                value.maxIters() == null ? base.maxIters() : value.maxIters());
    }

    public AiAgentDefinition definition(String type, Long userId) {
        return definition(type, get(type, userId).effective());
    }

    public String snapshot(String rootType, Long userId) {
        var snapshot = mapper.createObjectNode();
        if (rootType != null) {
            capture(rootType, userId, snapshot, new LinkedHashSet<>());
        }
        return snapshot.toString();
    }

    private void capture(String type, Long userId, ObjectNode snapshot, Set<String> visited) {
        if (!visited.add(type)) {
            return;
        }
        var configuration = get(type, userId);
        snapshot.set(type, mapper.valueToTree(configuration.overrides()));
        var definition = definition(type, configuration.effective());
        if (Integer.valueOf(1).equals(definition.getEnableTools())) {
            definition.getSubAgentTools().forEach(child -> capture(child.getRefAgentType(), userId, snapshot, visited));
        }
    }

    public AgentSettingsOverride snapshotOverride(String type, Map<String, String> variables) {
        // 未带用户覆盖的已受理快照沿用内置定义，不能在恢复时读取新设置改变旧任务。
        String snapshot = variables.get(SNAPSHOT_VARIABLE);
        if (snapshot == null) {
            return AgentSettingsOverride.inherit();
        }
        try {
            var overrides = mapper.readTree(snapshot).get(type);
            if (overrides == null) {
                throw new IllegalStateException("快照未包含该智能体的配置：" + type);
            }
            return mapper.treeToValue(overrides, AgentSettingsOverride.class);
        } catch (JsonProcessingException failure) {
            throw new IllegalStateException("智能体配置快照损坏", failure);
        }
    }

    public AiAgentDefinition fromSnapshot(String type, Map<String, String> variables) {
        return definition(type, merge(get(type, null).defaults(), snapshotOverride(type, variables)));
    }

    private AiAgentDefinition definition(String type, Values value) {
        var base = builtin(type);
        var children = base.getSubAgentTools() == null ? List.<AiAgentDefinition.SubAgentToolDef>of()
                : base.getSubAgentTools().stream()
                .filter(child -> value.subAgentToolNames().contains(child.getToolName()))
                .map(child -> mapper.convertValue(child, AiAgentDefinition.SubAgentToolDef.class)).toList();
        return AiAgentDefinition.builder().type(type).name(value.name()).systemPrompt(value.systemPrompt())
                .instructionTemplate(value.instructionTemplate()).defaultUserMessage(value.defaultUserMessage())
                .enableTools(value.toolsEnabled() ? 1 : 0).toolNames(value.toolNames()).subAgentTools(children)
                .defaultSkills(value.defaultSkills()).mcpToolNames(value.mcpToolNames()).maxIters(value.maxIters())
                .build();
    }

    private String text(String value) {
        return value == null ? "" : value;
    }
}
