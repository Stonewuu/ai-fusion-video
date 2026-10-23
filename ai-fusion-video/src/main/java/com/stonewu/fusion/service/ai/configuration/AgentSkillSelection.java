package com.stonewu.fusion.service.ai.configuration;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentScopeSkillRegistry;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentUserSkillService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * 对话、画布与子任务共用的显式 Skill 解析和正文注入。
 */
@Component
@RequiredArgsConstructor
public class AgentSkillSelection {
    public record Skill(String name, String displayName, String description, String content, String source) {
    }

    public record Option(String name, String displayName, String description, String source) {
    }

    private final AgentScopeSkillRegistry registry;
    private final AgentUserSkillService users;

    public List<Option> catalog(long userId) {
        return available(userId).values().stream()
                .map(skill -> new Option(skill.name(), skill.displayName(), skill.description(), skill.source()))
                .toList();
    }

    public List<Skill> resolve(List<String> names, Long userId) {
        return select(names, available(userId));
    }

    public List<Skill> freeze(Set<String> names, Long userId) {
        var available = available(userId);
        return names.stream().map(name -> {
            var skill = available.get(name);
            if (skill == null) {
                throw new BusinessException(400, "Skill 不可用：" + name + "。请重新选择或恢复智能体默认设置");
            }
            return skill;
        }).toList();
    }

    public List<Skill> select(List<String> names, List<Skill> frozen) {
        var catalog = new LinkedHashMap<String, Skill>();
        frozen.forEach(skill -> catalog.put(skill.name(), skill));
        return select(names, catalog);
    }

    private List<Skill> select(List<String> names, Map<String, Skill> available) {
        if (names == null || names.isEmpty()) {
            return List.of();
        }
        if (names.size() > 8) {
            throw new BusinessException(400, "每个任务最多选择 8 个 Skill");
        }
        List<Skill> selected = new ArrayList<>();
        var unique = new LinkedHashSet<String>();
        for (String name : names) {
            if (name == null || name.isBlank() || !unique.add(name)) {
                throw new BusinessException(400, "Skill 名称不能为空或重复");
            }
            var skill = available.get(name);
            if (skill == null) {
                throw new BusinessException(400, "Skill 不可用：" + name + "。请重新选择或恢复智能体默认设置");
            }
            selected.add(skill);
        }
        if (selected.stream().mapToInt(skill -> skill.content().length()).sum() > 128 * 1024) {
            throw new BusinessException(400, "所选 Skill 正文总长度超过 128 KB，请减少选择");
        }
        return List.copyOf(selected);
    }

    public List<String> names(List<String> defaults, List<String> selected, boolean inherit) {
        var names = new LinkedHashSet<String>();
        if (inherit && defaults != null) {
            names.addAll(defaults);
        }
        if (selected != null) {
            names.addAll(selected);
        }
        return List.copyOf(names);
    }

    public String prompt(List<Skill> selected) {
        if (selected.isEmpty()) {
            return "";
        }
        StringBuilder result = new StringBuilder("""
                ## 本任务启用的 Skills

                以下 Skill 由用户的智能体配置或本次任务选择启用。请理解并遵循正文要求；
                Skill 不会扩大工具权限、项目范围或生成预算。如要求冲突，请明确说明。
                """);
        for (var skill : selected) {
            result.append("\n### Skill: ").append(skill.name()).append("\n描述：")
                    .append(skill.description()).append("\n来源：").append(skill.source()).append("\n\n")
                    .append(skill.content()).append('\n');
        }
        return result.toString().strip();
    }

    private Map<String, Skill> available(Long userId) {
        var result = new LinkedHashMap<String, Skill>();
        for (var skill : registry.skills()) {
            result.put(skill.getName(),
                    new Skill(skill.getName(), skill.getName(), skill.getDescription(), skill.getSkillContent(),
                            skill.getSource()));
        }
        if (userId != null) {
            for (var skill : users.list(userId)) {
                result.put(skill.name(),
                        new Skill(skill.name(), skill.displayName(), skill.description(), skill.content(),
                                skill.source()));
            }
        }
        return result;
    }
}
