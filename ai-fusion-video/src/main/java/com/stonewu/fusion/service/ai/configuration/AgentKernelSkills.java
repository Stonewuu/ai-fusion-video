package com.stonewu.fusion.service.ai.configuration;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.ai.AiAgentDefinition;
import com.stonewu.fusion.controller.ai.vo.AiChatReqVO;
import com.stonewu.fusion.service.ai.AiAgentService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * 固定本次节点及子任务所需的 Skill 正文，恢复运行时不读取后来修改的文件。
 */
@Component
@RequiredArgsConstructor
public class AgentKernelSkills {
    public static final String SNAPSHOT = "agentSkillSnapshot";
    public static final String INHERIT_DEFAULTS = "inheritAgentSkillDefaults";
    public static final String ACTIVE_NAMES = "activeSkillNames";
    private final AgentSkillSelection skills;
    private final AiAgentService agents;
    private final ObjectMapper mapper;

    public String root(AiAgentDefinition definition, AiChatReqVO request, Map<String, String> variables, Long userId) {
        boolean inherit = !Boolean.FALSE.equals(request.getInheritAgentSkills());
        var active = skills.names(definition == null ? List.of() : definition.getDefaultSkills(),
                request.getEnabledSkills(), inherit);
        Set<String> required = new LinkedHashSet<>(active);
        if (inherit && definition != null) {
            defaults(definition, variables, required, new LinkedHashSet<>());
        }
        List<AgentSkillSelection.Skill> frozen;
        try {
            frozen = request.getAgentSkillSnapshot() == null ? skills.freeze(required, userId)
                    : mapper.readValue(request.getAgentSkillSnapshot(), new TypeReference<>() {
            });
        } catch (JsonProcessingException failure) {
            throw new IllegalStateException("画布 Skill 快照损坏", failure);
        }
        if (inherit && definition != null) {
            validateChildren(definition, variables, active, frozen, new LinkedHashSet<>());
        }
        try {
            variables.put(SNAPSHOT, mapper.writeValueAsString(frozen));
        } catch (JsonProcessingException failure) {
            throw new IllegalStateException("无法固定 Skill 正文", failure);
        }
        variables.put(INHERIT_DEFAULTS, Boolean.toString(inherit));
        variables.put(ACTIVE_NAMES, String.join("\n", active));
        return skills.prompt(skills.select(active, frozen));
    }

    public String child(AiAgentDefinition definition, Map<String, String> variables, Map<String, String> parent) {
        // 历史快照未启用节点 Skill，恢复它时仍保持原运行语义。
        if (!parent.containsKey(SNAPSHOT)) {
            variables.remove(SNAPSHOT);
            variables.remove(INHERIT_DEFAULTS);
            return "";
        }
        List<AgentSkillSelection.Skill> frozen;
        try {
            frozen = mapper.readValue(parent.get(SNAPSHOT), new TypeReference<>() {
            });
        } catch (JsonProcessingException failure) {
            throw new IllegalStateException("Skill 快照损坏", failure);
        }
        var inheritedNames = parent.get(ACTIVE_NAMES).isBlank() ? List.<String>of() : Arrays.asList(
                parent.get(ACTIVE_NAMES).split("\n"));
        boolean inheritDefaults = Boolean.parseBoolean(parent.get(INHERIT_DEFAULTS));
        var active = skills.names(inheritedNames, inheritDefaults ? definition.getDefaultSkills() : List.of(), true);
        variables.put(SNAPSHOT, parent.get(SNAPSHOT));
        variables.put(INHERIT_DEFAULTS, parent.get(INHERIT_DEFAULTS));
        variables.put(ACTIVE_NAMES, String.join("\n", active));
        return skills.prompt(skills.select(active, frozen));
    }

    private void defaults(AiAgentDefinition definition, Map<String, String> variables, Set<String> names,
                          Set<String> visited) {
        if (!visited.add(definition.getType())) {
            return;
        }
        names.addAll(definition.getDefaultSkills());
        if (Integer.valueOf(1).equals(definition.getEnableTools()) && definition.getSubAgentTools() != null) {
            definition.getSubAgentTools().forEach(
                    child -> defaults(agents.fromSnapshot(child.getRefAgentType(), variables), variables, names,
                            visited));
        }
    }

    private void validateChildren(AiAgentDefinition definition, Map<String, String> variables, List<String> inherited,
                                  List<AgentSkillSelection.Skill> frozen, Set<String> path) {
        if (!path.add(definition.getType())) {
            return;
        }
        var active = skills.names(inherited, definition.getDefaultSkills(), true);
        skills.select(active, frozen);
        if (Integer.valueOf(1).equals(definition.getEnableTools()) && definition.getSubAgentTools() != null) {
            definition.getSubAgentTools()
                    .forEach(child -> validateChildren(agents.fromSnapshot(child.getRefAgentType(), variables),
                            variables, active, frozen, new LinkedHashSet<>(path)));
        }
    }
}
