package com.stonewu.fusion.service.ai.agentscope.skill;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceBaseStore;
import io.agentscope.core.skill.AgentSkill;
import io.agentscope.core.skill.util.SkillUtil;
import io.agentscope.harness.agent.filesystem.remote.store.StoreItem;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class AgentUserSkillService {

    private static final int MAX_CONTENT_LENGTH = 256 * 1024;
    private static final int MAX_SKILLS_PER_USER = 64;
    private static final Pattern NAME_PATTERN = Pattern.compile("[a-z0-9][a-z0-9_-]{0,63}");

    private final AgentWorkspaceBaseStore workspaceStore;

    public List<UserSkill> list(long userId) {
        List<UserSkill> skills = new ArrayList<>();
        int offset = 0;
        while (true) {
            List<StoreItem> page = workspaceStore.search(namespace(userId), 100, offset);
            for (StoreItem item : page) {
                if (!isSkillDocument(item.key())) {
                    continue;
                }
                String markdown = fileContent(item);
                try {
                    AgentSkill skill = SkillUtil.createFrom(markdown, null, "workspace:user");
                    skills.add(new UserSkill(
                            skill.getSkillId(),
                            skill.getName(),
                            skill.getDescription(),
                            skill.getSkillContent(),
                            "workspace:user"));
                } catch (RuntimeException ignored) {
                    // Invalid external entries are ignored instead of breaking the whole catalog.
                }
            }
            if (page.size() < 100) {
                break;
            }
            offset += page.size();
        }
        return skills.stream().sorted(java.util.Comparator.comparing(UserSkill::name)).toList();
    }

    @Cacheable(value = "agentUserSkillCatalog", key = "#userId")
    public ArrayList<UserSkillSummary> catalog(long userId) {
        return list(userId).stream()
                .map(skill -> new UserSkillSummary(
                        skill.id(), skill.name(), skill.description(), skill.source()))
                .collect(Collectors.toCollection(ArrayList::new));
    }

    public UserSkill get(long userId, String name) {
        String safeName = requireName(name);
        return list(userId).stream()
                .filter(skill -> safeName.equals(skill.name()))
                .findFirst()
                .orElseThrow(() -> new BusinessException(404, "Skill 不存在"));
    }

    @CacheEvict(value = "agentUserSkillCatalog", key = "#userId")
    public UserSkill save(
            long userId,
            String originalName,
            String name,
            String description,
            String content) {
        String safeName = requireName(name);
        String safeDescription = requireText(description, "Skill 描述");
        String safeContent = requireText(content, "Skill 内容");
        if (safeContent.length() > MAX_CONTENT_LENGTH) {
            throw new BusinessException("Skill 内容不能超过 256 KB");
        }
        if ((originalName == null || originalName.isBlank())
                && list(userId).size() >= MAX_SKILLS_PER_USER) {
            throw new BusinessException("每个用户最多创建 64 个 Skill");
        }
        String oldName = originalName == null || originalName.isBlank()
                ? null
                : requireName(originalName);
        if (oldName == null && exists(userId, safeName)) {
            throw new BusinessException("同名 Skill 已存在");
        }
        if (oldName != null && !oldName.equals(safeName) && exists(userId, safeName)) {
            throw new BusinessException("同名 Skill 已存在");
        }
        workspaceStore.put(namespace(userId), skillKey(safeName), fileValue(
                toMarkdown(safeName, safeDescription, safeContent)));
        if (oldName != null && !oldName.equals(safeName)) {
            workspaceStore.delete(namespace(userId), skillKey(oldName));
        }
        AgentSkill saved = new AgentSkill(
                safeName, safeDescription, safeContent, Map.of(), "workspace:user");
        return new UserSkill(
                saved.getSkillId(),
                safeName,
                safeDescription,
                safeContent,
                "workspace:user");
    }

    @CacheEvict(value = "agentUserSkillCatalog", key = "#userId")
    public void delete(long userId, String name) {
        String safeName = requireName(name);
        workspaceStore.delete(namespace(userId), skillKey(safeName));
    }

    private boolean exists(long userId, String name) {
        return workspaceStore.get(namespace(userId), skillKey(name)) != null;
    }

    private List<String> namespace(long userId) {
        return List.of(
                "agents",
                AgentKernelSpecFactory.DEFAULT_AGENT_KEY,
                "users",
                String.valueOf(userId),
                "skills");
    }

    private String skillKey(String name) {
        return "/" + name + "/SKILL.md";
    }

    private boolean isSkillDocument(String key) {
        return key != null && key.matches("^/?[^/]+/SKILL\\.md$");
    }

    private String fileContent(StoreItem item) {
        Object content = item.value().get("content");
        return content == null ? "" : String.valueOf(content);
    }

    private Map<String, Object> fileValue(String markdown) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("content", markdown);
        value.put("encoding", "utf-8");
        String now = Instant.now().toString();
        value.put("created_at", now);
        value.put("modified_at", now);
        return value;
    }

    private String toMarkdown(String name, String description, String content) {
        return "---\nname: " + name + "\ndescription: \"" + yamlEscape(description)
                + "\"\n---\n" + content;
    }

    private String yamlEscape(String value) {
        return value.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\r", "")
                .replace("\n", "\\n");
    }

    private String requireName(String name) {
        String normalized = requireText(name, "Skill 名称").toLowerCase();
        if (!NAME_PATTERN.matcher(normalized).matches()) {
            throw new BusinessException("Skill 名称只能包含小写字母、数字、下划线和短横线，最长 64 位");
        }
        return normalized;
    }

    private String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new BusinessException(field + "不能为空");
        }
        return value.trim();
    }

    public record UserSkill(
            String id,
            String name,
            String description,
            String content,
            String source) {
    }

    public record UserSkillSummary(
            String id,
            String name,
            String description,
            String source) {
    }
}
