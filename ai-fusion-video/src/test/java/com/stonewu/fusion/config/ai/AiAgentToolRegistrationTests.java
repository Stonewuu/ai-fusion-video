package com.stonewu.fusion.config.ai;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.ToolPermissionRisk;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.tool.ToolResourceAccessGuard;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
class AiAgentToolRegistrationTests {

    @Autowired
    private AiAgentRegistry agentRegistry;

    @Autowired
    private List<ToolExecutor> toolExecutors;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void everyPlatformToolIsRegisteredExactlyOnceAndAssignedToAnAgent() {
        List<String> executorNames = toolExecutors.stream()
                .map(ToolExecutor::getToolName)
                .toList();
        Set<String> assignedNames = new LinkedHashSet<>();
        for (AiAgentDefinition agent : agentRegistry.getAll()) {
            if (agent.getToolNames() != null) {
                assignedNames.addAll(agent.getToolNames());
            }
        }

        assertThat(executorNames).doesNotHaveDuplicates();
        assertThat(assignedNames)
                .containsExactlyInAnyOrderElementsOf(executorNames);
    }

    @Test
    void everyReadOnlyDeclarationMatchesTheDefaultPermissionRisk() {
        for (ToolExecutor executor : toolExecutors) {
            if (executor.isReadOnly()) {
                assertThat(executor.getPermissionRisk(Map.of()))
                        .as(executor.getToolName())
                        .isEqualTo(ToolPermissionRisk.READ_ONLY);
            }
        }
    }

    @Test
    void everyPlatformToolExposesAValidAgentScopeSchema() {
        for (ToolExecutor executor : toolExecutors) {
            AgentScopeToolSchema.PreparedSchema schema = AgentScopeToolSchema.prepare(
                    objectMapper,
                    executor.getParametersSchema(),
                    executor.getToolName());
            assertThat(schema.canonicalJson()).as(executor.getToolName()).isNotBlank();
        }
    }

    @Test
    void lifecycleToolsAreRegisteredAndDeleteToolsAreHighRisk() {
        Set<String> lifecycleNames = Set.of(
                "save_project",
                "delete_project",
                "update_asset_item",
                "delete_asset_resource",
                "delete_script_child_resource",
                "update_storyboard_scene",
                "update_storyboard_item",
                "delete_storyboard_child_resource");

        assertThat(toolExecutors).extracting(ToolExecutor::getToolName).containsAll(lifecycleNames);
        assertThat(toolExecutors).extracting(ToolExecutor::getToolName)
                .doesNotContain("create_script", "save_storyboard", "delete_script_resource", "delete_storyboard_resource");
        toolExecutors.stream()
                .filter(tool -> tool.getToolName().startsWith("delete_"))
                .forEach(tool -> assertThat(tool.getPermissionRisk(Map.of()))
                        .as(tool.getToolName())
                        .isEqualTo(ToolPermissionRisk.HIGH_RISK));
    }

    @Test
    void deletionCapableSceneToolDeclaresDynamicHighRiskApproval() {
        ToolExecutor scenes = toolExecutors.stream()
                .filter(tool -> "manage_script_scenes".equals(tool.getToolName()))
                .findFirst()
                .orElseThrow();

        assertThat(scenes.mayRequireHighRiskApproval()).isTrue();
        assertThat(scenes.getPermissionRisk(Map.of("action", "add")))
                .isEqualTo(ToolPermissionRisk.EDIT);
        assertThat(scenes.getPermissionRisk(Map.of("action", "delete")))
                .isEqualTo(ToolPermissionRisk.HIGH_RISK);
    }

    @Test
    void overwriteSceneSaveRequiresHighRiskApproval() {
        ToolExecutor scenes = toolExecutors.stream()
                .filter(tool -> "save_script_scene_items".equals(tool.getToolName()))
                .findFirst()
                .orElseThrow();

        assertThat(scenes.mayRequireHighRiskApproval()).isTrue();
        assertThat(scenes.getPermissionRisk(Map.of("overwriteMode", false)))
                .isEqualTo(ToolPermissionRisk.EDIT);
        assertThat(scenes.getPermissionRisk(Map.of("overwriteMode", true)))
                .isEqualTo(ToolPermissionRisk.HIGH_RISK);
    }

    @Test
    void resourceIdBasedScriptAndStoryboardToolsUseTheSharedAccessGuard() {
        Set<String> guardedToolNames = Set.of(
                "get_script",
                "get_script_structure",
                "get_script_episode",
                "get_script_scene",
                "update_script",
                "update_script_info",
                "save_script_episode",
                "save_script_scene_items",
                "manage_script_scenes",
                "update_script_scene",
                "get_storyboard",
                "get_storyboard_scene_items",
                "save_storyboard_episode",
                "save_storyboard_scene_shots",
                "insert_storyboard_item",
                "update_storyboard_item_frame",
                "update_storyboard_item_video");

        toolExecutors.stream()
                .filter(tool -> guardedToolNames.contains(tool.getToolName()))
                .forEach(tool -> assertThat(tool.getClass().getDeclaredFields())
                        .as(tool.getToolName())
                        .anyMatch(field -> field.getType().equals(ToolResourceAccessGuard.class)));
        assertThat(toolExecutors.stream()
                .map(ToolExecutor::getToolName)
                .filter(guardedToolNames::contains))
                .containsExactlyInAnyOrderElementsOf(guardedToolNames);
    }
}
