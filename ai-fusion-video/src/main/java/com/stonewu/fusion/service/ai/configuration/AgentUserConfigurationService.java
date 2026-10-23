package com.stonewu.fusion.service.ai.configuration;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.ai.AiAgentRegistry;
import com.stonewu.fusion.service.ai.AiToolConfigService;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentScopeMcpRegistry;
import jakarta.validation.Validator;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

@Service
@RequiredArgsConstructor
public class AgentUserConfigurationService {
    public record Summary(String type, String builtinName, String name, boolean customized, long revision,
                          boolean toolsEnabled, List<String> defaultSkills, int maxIters) {
    }

    public record ToolOption(String name, String label, String description, String kind, boolean readOnly,
                             String category) {
    }

    public record Detail(AgentDefinitionConfiguration.Resolved configuration, List<ToolOption> tools,
                         List<AgentSkillSelection.Option> skills) {
    }

    private final AiAgentRegistry registry;
    private final AgentDefinitionConfiguration definitions;
    private final AgentUserOverrideRepository repository;
    private final AiToolConfigService tools;
    private final AgentScopeMcpRegistry mcp;
    private final AgentSkillSelection skills;
    private final ObjectMapper mapper;
    private final Validator validator;

    public List<Summary> list(long userId) {
        return registry.getAll().stream().map(base -> {
            var value = definitions.get(base.getType(), userId);
            return new Summary(base.getType(), base.getName(), value.effective().name(),
                    !json(value.overrides()).isEmpty(), value.revision(),
                    value.effective().toolsEnabled(), value.effective().defaultSkills(), value.effective().maxIters());
        }).toList();
    }

    public Detail detail(long userId, String type) {
        return new Detail(definitions.get(type, userId), toolOptions(userId, type), skills.catalog(userId));
    }

    @Transactional
    @CacheEvict(value = AgentUserOverrideRepository.CACHE, key = "#userId + ':' + #type")
    public long save(long userId, String type, long expectedRevision, AgentSettingsOverride overrides) {
        if (overrides == null || expectedRevision < 0) {
            throw new BusinessException(400, "智能体配置或版本无效");
        }
        var violations = validator.validate(overrides);
        if (!violations.isEmpty()) {
            throw new BusinessException(400, "请检查配置长度、选择数量和步数设置");
        }
        var current = definitions.get(type, userId);
        var effective = definitions.merge(current.defaults(), overrides);
        if (effective.name().isBlank() || effective.systemPrompt().isBlank()) {
            throw new BusinessException(400, "智能体名称和系统提示词不能为空");
        }
        var options = toolOptions(userId, type);
        validateNames(overrides.toolNames(), options, "platform", "平台工具");
        validateNames(overrides.subAgentToolNames(), options, "agent", "子任务工具");
        validateNames(overrides.mcpToolNames(), options, "mcp", "MCP 工具");
        skills.resolve(effective.defaultSkills(), userId);
        try {
            return repository.save(userId, type, expectedRevision, mapper.writeValueAsString(json(overrides)));
        } catch (JsonProcessingException failure) {
            throw new IllegalStateException("无法保存智能体配置", failure);
        }
    }

    private List<ToolOption> toolOptions(long userId, String type) {
        var base = definitions.builtin(type);
        var result = new ArrayList<ToolOption>();
        tools.getEnabledTools().forEach(tool -> result.add(new ToolOption(tool.getToolName(), tool.getDisplayName(),
                tool.getToolDescription(), "platform", tool.isReadOnly(),
                AgentToolCategories.category(tool.getToolName()))));
        if (base.getSubAgentTools() != null) {
            base.getSubAgentTools().forEach(tool -> result.add(new ToolOption(tool.getToolName(),
                    tool.getDisplayName(), tool.getDescription(), "agent", false, "子任务委派")));
        }
        mcp.catalogForAgent(type, userId)
                .forEach(tool -> result.add(new ToolOption(tool.toolName(), tool.serverName() + " · " + tool.toolName(),
                        tool.description(), "mcp", tool.readOnly(), "MCP 外部服务")));
        return List.copyOf(result);
    }

    private void validateNames(List<String> selected, List<ToolOption> catalog, String kind, String label) {
        if (selected == null) {
            return;
        }
        Set<String> unique = new LinkedHashSet<>();
        Set<String> available = new LinkedHashSet<>();
        catalog.stream().filter(tool -> tool.kind().equals(kind)).forEach(tool -> available.add(tool.name()));
        for (String name : selected) {
            if (name == null || name.isBlank() || !unique.add(name) || !available.contains(name)) {
                throw new BusinessException(400, label + "不可用或重复：" + name + "，请刷新选项后重新选择");
            }
        }
    }

    private ObjectNode json(AgentSettingsOverride value) {
        ObjectNode object = mapper.valueToTree(value);
        var absent = new ArrayList<String>();
        object.properties().forEach(entry -> {
            if (entry.getValue().isNull()) {
                absent.add(entry.getKey());
            }
        });
        object.remove(absent);
        return object;
    }
}
