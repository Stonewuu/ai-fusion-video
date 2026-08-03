package com.stonewu.fusion.service.ai.agentscope.skill;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceBaseStore;
import io.agentscope.harness.agent.filesystem.remote.store.StoreItem;
import io.agentscope.harness.agent.filesystem.remote.store.BaseStore;
import io.agentscope.harness.agent.filesystem.AbstractFilesystem;
import io.agentscope.harness.agent.filesystem.spec.RemoteFilesystemSpec;
import io.agentscope.harness.agent.IsolationScope;
import io.agentscope.core.agent.RuntimeContext;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.Map;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.AbstractMap;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class AgentUserSkillServiceTests {

    @TempDir
    Path temporaryDirectory;

    @Test
    void savesSkillInTheOfficialPerUserWorkspaceNamespace() {
        AgentWorkspaceBaseStore store = mock(AgentWorkspaceBaseStore.class);
        AgentUserSkillService service = new AgentUserSkillService(store);
        when(store.get(anyList(), eq("/story-review/SKILL.md"))).thenReturn(null);

        AgentUserSkillService.UserSkill saved = service.save(
                42L,
                null,
                "story-review",
                "故事结构检查",
                "检查故事结构",
                "# 步骤\n\n检查冲突与节奏。"
        );

        ArgumentCaptor<List<String>> namespace = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<Map<String, Object>> skillValue = ArgumentCaptor.forClass(Map.class);
        ArgumentCaptor<Map<String, Object>> metadataValue = ArgumentCaptor.forClass(Map.class);
        verify(store).put(namespace.capture(), eq("/story-review/SKILL.md"), skillValue.capture());
        verify(store).put(anyList(), eq("/story-review/.fusion-skill.json"), metadataValue.capture());
        assertThat(namespace.getValue()).containsExactly(
                "agents", "ai_assistant_agent", "users", "42", "skills");
        assertThat(skillValue.getValue().get("content").toString())
                .contains("name: story-review", "description: 检查故事结构");
        assertThat(metadataValue.getValue().get("display_name")).isEqualTo("故事结构检查");
        assertThat(saved.name()).isEqualTo("story-review");
        assertThat(saved.displayName()).isEqualTo("故事结构检查");
    }

    @Test
    void listsOnlyValidSkillDocuments() {
        AgentWorkspaceBaseStore store = mock(AgentWorkspaceBaseStore.class);
        AgentUserSkillService service = new AgentUserSkillService(store);
        StoreItem item = new StoreItem(
                "/story-review/SKILL.md",
                Map.of("content", """
                        ---
                        name: story-review
                        description: 检查故事结构
                        ---
                        # 步骤
                        检查冲突与节奏。
                        """),
                1L);
        when(store.search(anyList(), eq(100), eq(0))).thenReturn(List.of(item));
        when(store.get(anyList(), eq("/story-review/.fusion-skill.json"))).thenReturn(
                new StoreItem(
                        "/story-review/.fusion-skill.json",
                        Map.of("display_name", "故事结构检查"),
                        1L));

        assertThat(service.list(42L))
                .singleElement()
                .satisfies(skill -> {
                    assertThat(skill.name()).isEqualTo("story-review");
                    assertThat(skill.displayName()).isEqualTo("故事结构检查");
                    assertThat(skill.content()).contains("检查冲突与节奏");
                });
    }

    @Test
    void keepsLegacySkillVisibleForEditingButExcludesItFromReferenceCatalog() {
        AgentWorkspaceBaseStore store = mock(AgentWorkspaceBaseStore.class);
        AgentUserSkillService service = new AgentUserSkillService(store);
        when(store.search(anyList(), eq(100), eq(0))).thenReturn(List.of(skillItem("story-review")));
        when(store.get(anyList(), eq("/story-review/.fusion-skill.json"))).thenReturn(null);

        assertThat(service.list(42L))
                .singleElement()
                .satisfies(skill -> {
                    assertThat(skill.name()).isEqualTo("story-review");
                    assertThat(skill.displayName()).isNull();
                });
        assertThat(service.catalog(42L)).isEmpty();
    }

    @Test
    void renamingAndDeletingSkillAlsoMovesItsDisplayNameMetadata() {
        AgentWorkspaceBaseStore store = mock(AgentWorkspaceBaseStore.class);
        AgentUserSkillService service = new AgentUserSkillService(store);
        when(store.get(anyList(), eq("/story-a/SKILL.md"))).thenReturn(skillItem("story-a"));
        when(store.get(anyList(), eq("/story-b/SKILL.md"))).thenReturn(null);
        when(store.search(anyList(), eq(100), eq(0))).thenReturn(List.of(
                skillItem("story-a"),
                metadataItem("story-a")));

        service.save(42L, "story-a", "story-b", "故事检查", "检查故事", "# 步骤");

        verify(store).put(anyList(), eq("/story-b/SKILL.md"), anyMap());
        verify(store).put(anyList(), eq("/story-b/.fusion-skill.json"), anyMap());
        verify(store).delete(anyList(), eq("/story-a/SKILL.md"));
        verify(store).delete(anyList(), eq("/story-a/.fusion-skill.json"));

        when(store.search(anyList(), eq(100), eq(0))).thenReturn(List.of(
                skillItem("story-b"),
                metadataItem("story-b")));
        service.delete(42L, "story-b");

        verify(store).delete(anyList(), eq("/story-b/SKILL.md"));
        verify(store).delete(anyList(), eq("/story-b/.fusion-skill.json"));
    }

    @Test
    void rejectsDisplayNameLongerThanSixtyFourCharacters() {
        AgentWorkspaceBaseStore store = mock(AgentWorkspaceBaseStore.class);
        AgentUserSkillService service = new AgentUserSkillService(store);

        assertThatThrownBy(() -> service.save(
                42L, null, "story-review", "中".repeat(65), "检查故事", "# 步骤"))
                .isInstanceOf(BusinessException.class)
                .hasMessage("Skill 显示名称不能超过 64 个字符");
        verifyNoInteractions(store);
    }

    @Test
    void serviceNamespaceMatchesAgentScopeRemoteFilesystemRouting() {
        CapturingStore store = new CapturingStore();
        AbstractFilesystem filesystem = new RemoteFilesystemSpec(store)
                .isolationScope(IsolationScope.USER)
                .toFilesystem(
                        temporaryDirectory,
                        "ai_assistant_agent",
                        runtimeContext -> List.of("local"));
        RuntimeContext context = RuntimeContext.builder()
                .userId("42")
                .sessionId("session")
                .build();

        filesystem.uploadFiles(context, List.of(new AbstractMap.SimpleImmutableEntry<>(
                "skills/story-review/SKILL.md",
                "content".getBytes(StandardCharsets.UTF_8))));

        assertThat(store.namespace).containsExactly(
                "agents", "ai_assistant_agent", "users", "42", "skills");
        assertThat(store.key).isEqualTo("/story-review/SKILL.md");
    }

    private StoreItem skillItem(String name) {
        return new StoreItem(
                "/" + name + "/SKILL.md",
                Map.of("content", """
                        ---
                        name: %s
                        description: 检查故事结构
                        ---
                        # 步骤
                        检查冲突与节奏。
                        """.formatted(name)),
                1L);
    }

    private StoreItem metadataItem(String name) {
        return new StoreItem(
                "/" + name + "/.fusion-skill.json",
                Map.of("display_name", name),
                1L);
    }

    private static final class CapturingStore implements BaseStore {

        private List<String> namespace;
        private String key;

        @Override
        public StoreItem get(List<String> namespace, String key) {
            return null;
        }

        @Override
        public void put(List<String> namespace, String key, Map<String, Object> value) {
            this.namespace = List.copyOf(namespace);
            this.key = key;
        }

        @Override
        public List<StoreItem> search(List<String> namespace, int limit, int offset) {
            return List.of();
        }

        @Override
        public void delete(List<String> namespace, String key) {
        }
    }
}
