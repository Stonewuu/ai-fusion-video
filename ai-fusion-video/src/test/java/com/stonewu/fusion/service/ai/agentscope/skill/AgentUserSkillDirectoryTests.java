package com.stonewu.fusion.service.ai.agentscope.skill;

import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceBaseStore;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.harness.agent.IsolationScope;
import io.agentscope.harness.agent.filesystem.AbstractFilesystem;
import io.agentscope.harness.agent.filesystem.spec.RemoteFilesystemSpec;
import io.agentscope.harness.agent.skill.SkillResources;
import io.agentscope.harness.agent.skill.WorkspaceSkillRepository;
import io.agentscope.harness.agent.filesystem.remote.store.StoreItem;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.ArgumentCaptor;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentUserSkillDirectoryTests {

    @TempDir
    Path temporaryDirectory;

    @Test
    void storesBinaryAssetsAsBase64WithoutLosingBytes() {
        AgentWorkspaceBaseStore store = mock(AgentWorkspaceBaseStore.class);
        when(store.search(anyList(), eq(100), eq(0))).thenReturn(List.of());
        AgentUserSkillService service = new AgentUserSkillService(store);
        Map<String, byte[]> files = new LinkedHashMap<>();
        files.put("SKILL.md", skillMarkdown());
        byte[] binary = new byte[]{(byte) 0xff, 0x00, (byte) 0x81, 0x42};
        files.put("assets/template.bin", binary);

        service.replaceImportedDirectory(42L, "binary-skill", "Binary Skill", files);

        ArgumentCaptor<Map<String, Object>> value = ArgumentCaptor.forClass(Map.class);
        verify(store).put(anyList(), eq("/binary-skill/assets/template.bin"), value.capture());
        assertThat(value.getValue().get("encoding")).isEqualTo("base64");
        assertThat(Base64.getDecoder().decode(String.valueOf(value.getValue().get("content"))))
                .containsExactly(binary);
    }

    @Test
    void deletingSkillDeletesEveryFileInItsDirectory() {
        AgentWorkspaceBaseStore store = mock(AgentWorkspaceBaseStore.class);
        when(store.search(anyList(), eq(100), eq(0))).thenReturn(List.of(
                new StoreItem("/full-skill/SKILL.md", Map.of("content", "skill"), 1L),
                new StoreItem("/full-skill/references/guide.md", Map.of("content", "guide"), 1L),
                new StoreItem("/full-skill/assets/template.txt", Map.of("content", "template"), 1L),
                new StoreItem("/other-skill/SKILL.md", Map.of("content", "other"), 1L)));
        AgentUserSkillService service = new AgentUserSkillService(store);

        service.delete(42L, "full-skill");

        verify(store).delete(anyList(), eq("/full-skill/SKILL.md"));
        verify(store).delete(anyList(), eq("/full-skill/references/guide.md"));
        verify(store).delete(anyList(), eq("/full-skill/assets/template.txt"));
    }

    @Test
    void editingImportedSkillPreservesOptionalFrontmatterAndDirectoryResources() {
        AgentWorkspaceBaseStore store = mock(AgentWorkspaceBaseStore.class);
        StoreItem skill = new StoreItem(
                "/editable-skill/SKILL.md",
                Map.of("content", """
                        ---
                        name: editable-skill
                        description: Old description
                        license: Apache-2.0
                        compatibility: Requires git
                        metadata:
                          author: example
                        ---
                        # Old instructions
                        """, "encoding", "utf-8"),
                1L);
        StoreItem reference = new StoreItem(
                "/editable-skill/references/guide.md",
                Map.of("content", "guide", "encoding", "utf-8"),
                1L);
        when(store.get(anyList(), eq("/editable-skill/SKILL.md"))).thenReturn(skill);
        when(store.search(anyList(), eq(100), eq(0))).thenReturn(List.of(skill, reference));
        AgentUserSkillService service = new AgentUserSkillService(store);

        service.save(
                42L,
                "editable-skill",
                "editable-skill",
                "Editable Skill",
                "New description",
                "# New instructions");

        ArgumentCaptor<Map<String, Object>> markdownValue = ArgumentCaptor.forClass(Map.class);
        verify(store).put(anyList(), eq("/editable-skill/SKILL.md"), markdownValue.capture());
        assertThat(markdownValue.getValue().get("content").toString())
                .contains(
                        "description: New description",
                        "license: Apache-2.0",
                        "compatibility: Requires git",
                        "author: example",
                        "# New instructions");
        verify(store).put(anyList(), eq("/editable-skill/references/guide.md"),
                org.mockito.ArgumentMatchers.anyMap());
    }

    @Test
    void importedReferencesAndBinaryAssetsAreReadableThroughAgentScopeWorkspaceRepository() {
        AgentWorkspaceBaseStore store = mock(AgentWorkspaceBaseStore.class);
        Map<String, StoreItem> entries = new LinkedHashMap<>();
        doAnswer(invocation -> {
            String key = invocation.getArgument(1);
            Map<String, Object> value = invocation.getArgument(2);
            entries.put(key, new StoreItem(key, value, 1L));
            return null;
        }).when(store).put(anyList(), anyString(), org.mockito.ArgumentMatchers.anyMap());
        doAnswer(invocation -> {
            entries.remove(invocation.getArgument(1, String.class));
            return null;
        }).when(store).delete(anyList(), anyString());
        when(store.get(anyList(), anyString())).thenAnswer(invocation ->
                entries.get(invocation.getArgument(1, String.class)));
        when(store.search(anyList(), anyInt(), anyInt())).thenAnswer(invocation -> {
            int limit = invocation.getArgument(1, Integer.class);
            int offset = invocation.getArgument(2, Integer.class);
            List<StoreItem> ordered = entries.values().stream()
                    .sorted(Comparator.comparing(StoreItem::key))
                    .toList();
            if (offset >= ordered.size()) return List.of();
            return new ArrayList<>(ordered.subList(offset, Math.min(offset + limit, ordered.size())));
        });
        AgentUserSkillService service = new AgentUserSkillService(store);
        Map<String, byte[]> files = new LinkedHashMap<>();
        files.put("SKILL.md", skillMarkdown());
        files.put("references/guide.md", "reference body".getBytes(StandardCharsets.UTF_8));
        byte[] binary = new byte[]{(byte) 0xff, 0x00, 0x42};
        files.put("assets/template.bin", binary);
        service.replaceImportedDirectory(42L, "binary-skill", "Binary Skill", files);

        RuntimeContext context = RuntimeContext.builder().userId("42").sessionId("session").build();
        AbstractFilesystem filesystem = new RemoteFilesystemSpec(store)
                .isolationScope(IsolationScope.USER)
                .toFilesystem(temporaryDirectory, "ai_assistant_agent", ignored -> List.of("local"));
        WorkspaceSkillRepository repository = new WorkspaceSkillRepository(
                filesystem, "skills", () -> context);
        SkillResources resources = repository.resourcesFor("binary-skill", context);

        assertThat(repository.getSkill("binary-skill")).isNotNull();
        assertThat(resources.read("references/guide.md")).contains("reference body");
        assertThat(resources.readBinary("assets/template.bin")).hasValueSatisfying(value ->
                assertThat(value).containsExactly(binary));
    }

    private byte[] skillMarkdown() {
        return """
                ---
                name: binary-skill
                description: Use when binary templates are needed
                ---
                # Instructions
                Use the bundled template.
                """.getBytes(StandardCharsets.UTF_8);
    }
}
