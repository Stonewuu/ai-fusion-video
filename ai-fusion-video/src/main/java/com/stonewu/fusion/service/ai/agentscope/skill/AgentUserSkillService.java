package com.stonewu.fusion.service.ai.agentscope.skill;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceBaseStore;
import io.agentscope.core.skill.AgentSkill;
import io.agentscope.core.skill.util.MarkdownSkillParser;
import io.agentscope.core.skill.util.SkillUtil;
import io.agentscope.harness.agent.filesystem.remote.store.StoreItem;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class AgentUserSkillService {

    static final int MAX_CONTENT_LENGTH = 256 * 1024;
    static final int MAX_SKILLS_PER_USER = 64;
    static final int MAX_DISPLAY_NAME_LENGTH = 64;
    static final int MAX_DESCRIPTION_LENGTH = 1024;
    static final String METADATA_FILENAME = ".fusion-skill.json";
    static final Pattern STANDARD_NAME_PATTERN = Pattern.compile("[a-z0-9]+(?:-[a-z0-9]+)*");

    private static final String DISPLAY_NAME_FIELD = "display_name";
    private static final Pattern LEGACY_NAME_PATTERN = Pattern.compile("[a-z0-9][a-z0-9_-]{0,63}");

    private final AgentWorkspaceBaseStore workspaceStore;

    public List<UserSkill> list(long userId) {
        List<UserSkill> skills = new ArrayList<>();
        for (StoreItem item : workspaceItems(userId)) {
            if (!isSkillDocument(item.key())) {
                continue;
            }
            String markdown = fileContent(item);
            try {
                AgentSkill skill = SkillUtil.createFrom(markdown, null, "workspace:user");
                skills.add(new UserSkill(
                        skill.getSkillId(),
                        skill.getName(),
                        displayName(userId, skill.getName()),
                        skill.getDescription(),
                        skill.getSkillContent(),
                        "workspace:user"));
            } catch (RuntimeException ignored) {
                // Invalid external entries are ignored instead of breaking the whole catalog.
            }
        }
        return skills.stream().sorted(Comparator.comparing(UserSkill::name)).toList();
    }

    @Cacheable(value = "agentUserSkillCatalog", key = "#userId")
    public ArrayList<UserSkillSummary> catalog(long userId) {
        return list(userId).stream()
                .filter(skill -> skill.displayName() != null)
                .map(skill -> new UserSkillSummary(
                        skill.id(), skill.name(), skill.displayName(),
                        skill.description(), skill.source()))
                .collect(Collectors.toCollection(ArrayList::new));
    }

    public UserSkill get(long userId, String name) {
        String safeName = requireLookupName(name);
        return list(userId).stream()
                .filter(skill -> safeName.equals(skill.name()))
                .findFirst()
                .orElseThrow(() -> new BusinessException(404, "Skill 不存在"));
    }

    @Transactional
    @CacheEvict(value = "agentUserSkillCatalog", key = "#userId")
    public UserSkill save(
            long userId,
            String originalName,
            String name,
            String displayName,
            String description,
            String content) {
        String oldName = originalName == null || originalName.isBlank()
                ? null
                : requireLookupName(originalName);
        String safeName = requireSaveName(oldName, name);
        String safeDisplayName = requireDisplayName(displayName);
        String safeDescription = requireText(description, "Skill 描述");
        String safeContent = requireText(content, "Skill 内容");
        if (codePointLength(safeDescription) > MAX_DESCRIPTION_LENGTH) {
            throw new BusinessException("Skill 描述不能超过 1024 个字符");
        }
        if (safeContent.getBytes(StandardCharsets.UTF_8).length > MAX_CONTENT_LENGTH) {
            throw new BusinessException("Skill 内容不能超过 256 KB");
        }
        if (oldName == null && skillCount(userId) >= MAX_SKILLS_PER_USER) {
            throw new BusinessException("每个用户最多创建 64 个 Skill");
        }
        if (oldName == null && exists(userId, safeName)) {
            throw new BusinessException("同名 Skill 已存在");
        }
        if (oldName != null && !exists(userId, oldName)) {
            throw new BusinessException(404, "Skill 不存在");
        }
        if (oldName != null && !oldName.equals(safeName) && exists(userId, safeName)) {
            throw new BusinessException("同名 Skill 已存在");
        }

        Map<String, byte[]> files = oldName == null
                ? new LinkedHashMap<>()
                : readDirectoryFiles(userId, oldName);
        Map<String, Object> metadata = existingMetadata(userId, oldName);
        metadata.put("name", safeName);
        metadata.put("description", safeDescription);
        String markdown = MarkdownSkillParser.generate(metadata, safeContent);
        files.put("SKILL.md", markdown.getBytes(StandardCharsets.UTF_8));
        replaceDirectory(userId, oldName, safeName, safeDisplayName, files);

        AgentSkill saved = SkillUtil.createFrom(markdown, null, "workspace:user");
        return new UserSkill(
                saved.getSkillId(),
                safeName,
                safeDisplayName,
                safeDescription,
                safeContent,
                "workspace:user");
    }

    @Transactional
    @CacheEvict(value = "agentUserSkillCatalog", key = "#userId")
    public void delete(long userId, String name) {
        deleteDirectory(userId, requireLookupName(name));
    }

    boolean exists(long userId, String name) {
        return workspaceStore.get(namespace(userId), skillKey(name)) != null;
    }

    int skillCount(long userId) {
        return list(userId).size();
    }

    String requireDisplayName(String displayName) {
        String safeDisplayName = requireText(displayName, "Skill 显示名称");
        if (codePointLength(safeDisplayName) > MAX_DISPLAY_NAME_LENGTH) {
            throw new BusinessException("Skill 显示名称不能超过 64 个字符");
        }
        return safeDisplayName;
    }

    String requireStandardName(String name) {
        String value = requireText(name, "Skill 名称");
        if (codePointLength(value) > 64 || !STANDARD_NAME_PATTERN.matcher(value).matches()) {
            throw new BusinessException(
                    "Skill 名称只能包含小写字母、数字和短横线，不能以短横线开头或结尾，也不能包含连续短横线，最长 64 位");
        }
        return value;
    }

    void replaceImportedDirectory(
            long userId,
            String name,
            String displayName,
            Map<String, byte[]> files) {
        String safeName = requireStandardName(name);
        String safeDisplayName = requireDisplayName(displayName);
        if (files == null || !files.containsKey("SKILL.md")) {
            throw new BusinessException("导入的 Skill 缺少 SKILL.md");
        }
        replaceDirectory(userId, safeName, safeName, safeDisplayName, files);
    }

    private String requireSaveName(String oldName, String requestedName) {
        String value = requireText(requestedName, "Skill 名称");
        if (oldName != null && oldName.equals(value.toLowerCase(Locale.ROOT))) {
            return oldName;
        }
        return requireStandardName(value);
    }

    private String requireLookupName(String name) {
        String normalized = requireText(name, "Skill 名称").toLowerCase(Locale.ROOT);
        if (!LEGACY_NAME_PATTERN.matcher(normalized).matches()) {
            throw new BusinessException("Skill 名称格式不正确");
        }
        return normalized;
    }

    private Map<String, Object> existingMetadata(long userId, String name) {
        if (name == null) {
            return new LinkedHashMap<>();
        }
        StoreItem item = workspaceStore.get(namespace(userId), skillKey(name));
        if (item == null) {
            return new LinkedHashMap<>();
        }
        MarkdownSkillParser.ParsedMarkdown parsed = MarkdownSkillParser.parse(fileContent(item));
        return new LinkedHashMap<>(parsed.getMetadata());
    }

    private Map<String, byte[]> readDirectoryFiles(long userId, String name) {
        Map<String, byte[]> files = new LinkedHashMap<>();
        String prefix = directoryPrefix(name);
        for (StoreItem item : workspaceItems(userId)) {
            String key = normalizeStoreKey(item.key());
            if (!key.startsWith(prefix)) {
                continue;
            }
            String relativePath = key.substring(prefix.length());
            if (METADATA_FILENAME.equals(relativePath)) {
                continue;
            }
            files.put(relativePath, fileBytes(item));
        }
        return files;
    }

    private void replaceDirectory(
            long userId,
            String oldName,
            String newName,
            String displayName,
            Map<String, byte[]> files) {
        List<String> namespace = namespace(userId);
        String targetPrefix = directoryPrefix(newName);
        Set<String> desiredKeys = new LinkedHashSet<>();
        files.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .forEach(entry -> {
                    String key = targetPrefix + entry.getKey();
                    desiredKeys.add(key);
                    workspaceStore.put(namespace, key, fileValue(entry.getValue()));
                });
        String metadataKey = metadataKey(newName);
        desiredKeys.add(metadataKey);
        workspaceStore.put(namespace, metadataKey, metadataValue(displayName));

        for (StoreItem item : workspaceItems(userId)) {
            String key = normalizeStoreKey(item.key());
            if (key.startsWith(targetPrefix) && !desiredKeys.contains(key)) {
                workspaceStore.delete(namespace, key);
            }
        }
        if (oldName != null && !oldName.equals(newName)) {
            deleteDirectory(userId, oldName);
        }
    }

    private void deleteDirectory(long userId, String name) {
        List<String> namespace = namespace(userId);
        String prefix = directoryPrefix(name);
        List<String> keys = workspaceItems(userId).stream()
                .map(StoreItem::key)
                .map(this::normalizeStoreKey)
                .filter(key -> key.startsWith(prefix))
                .toList();
        keys.forEach(key -> workspaceStore.delete(namespace, key));
    }

    private List<StoreItem> workspaceItems(long userId) {
        List<StoreItem> items = new ArrayList<>();
        int offset = 0;
        while (true) {
            List<StoreItem> page = workspaceStore.search(namespace(userId), 100, offset);
            items.addAll(page);
            if (page.size() < 100) {
                return items;
            }
            offset += page.size();
        }
    }

    private List<String> namespace(long userId) {
        return List.of(
                "agents",
                AgentKernelSpecFactory.DEFAULT_AGENT_KEY,
                "users",
                String.valueOf(userId),
                "skills");
    }

    private String directoryPrefix(String name) {
        return "/" + name + "/";
    }

    private String skillKey(String name) {
        return directoryPrefix(name) + "SKILL.md";
    }

    private String metadataKey(String name) {
        return directoryPrefix(name) + METADATA_FILENAME;
    }

    private boolean isSkillDocument(String key) {
        return key != null && key.matches("^/?[^/]+/SKILL\\.md$");
    }

    private String normalizeStoreKey(String key) {
        return key != null && key.startsWith("/") ? key : "/" + key;
    }

    private String fileContent(StoreItem item) {
        byte[] bytes = fileBytes(item);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private byte[] fileBytes(StoreItem item) {
        Object content = item.value().get("content");
        if (content == null) {
            return new byte[0];
        }
        String text = String.valueOf(content);
        if ("base64".equals(item.value().get("encoding"))) {
            try {
                return Base64.getMimeDecoder().decode(text);
            } catch (IllegalArgumentException failure) {
                throw new BusinessException("Skill 资源的 Base64 内容无效");
            }
        }
        return text.getBytes(StandardCharsets.UTF_8);
    }

    private Map<String, Object> fileValue(byte[] bytes) {
        Map<String, Object> value = new LinkedHashMap<>();
        try {
            String text = StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(bytes))
                    .toString();
            value.put("content", text);
            value.put("encoding", "utf-8");
        } catch (CharacterCodingException binary) {
            value.put("content", Base64.getEncoder().encodeToString(bytes));
            value.put("encoding", "base64");
        }
        String now = Instant.now().toString();
        value.put("created_at", now);
        value.put("modified_at", now);
        return value;
    }

    private Map<String, Object> metadataValue(String displayName) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put(DISPLAY_NAME_FIELD, displayName);
        String now = Instant.now().toString();
        value.put("created_at", now);
        value.put("modified_at", now);
        return value;
    }

    private String displayName(long userId, String name) {
        StoreItem metadata = workspaceStore.get(namespace(userId), metadataKey(name));
        if (metadata == null) {
            return null;
        }
        Object value = metadata.value().get(DISPLAY_NAME_FIELD);
        if (!(value instanceof String text) || text.isBlank()) {
            return null;
        }
        return text.trim();
    }

    private int codePointLength(String value) {
        return value.codePointCount(0, value.length());
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
            String displayName,
            String description,
            String content,
            String source) {
    }

    public record UserSkillSummary(
            String id,
            String name,
            String displayName,
            String description,
            String source) {
    }
}
