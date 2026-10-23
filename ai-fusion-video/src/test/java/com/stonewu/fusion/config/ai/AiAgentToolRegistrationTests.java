package com.stonewu.fusion.config.ai;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.configuration.AgentToolCategories;
import com.stonewu.fusion.service.ai.ToolPermissionRisk;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
@Testcontainers(disabledWithoutDocker = false)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class AiAgentToolRegistrationTests {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "agent-tool-registration-" + UUID.randomUUID()).toAbsolutePath();

    @DynamicPropertySource static void isolatedProperties(DynamicPropertyRegistry properties) {
        properties.add("spring.datasource.url", MYSQL::getJdbcUrl);
        properties.add("spring.datasource.username", MYSQL::getUsername);
        properties.add("spring.datasource.password", MYSQL::getPassword);
        properties.add("spring.data.redis.host", REDIS::getHost);
        properties.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379));
        properties.add("spring.data.redis.password", () -> "");
        properties.add("app.storage.local-base-path", FILES::toString);
        properties.add("app.agent-workspace.local-base-path", () -> FILES.resolve("workspace").toString());
    }

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
        executorNames.forEach(name -> assertThat(AgentToolCategories.category(name)).as(name).isNotBlank());
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
    void resourceIdBasedScriptAndStoryboardToolsRejectUntrustedInvocationIdentity() throws Exception {
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

        for (ToolExecutor tool : toolExecutors) {
            if (!guardedToolNames.contains(tool.getToolName())) continue;
            String response = tool.execute("{}", ToolExecutionContext.builder().userId(42L).build());
            assertThat(objectMapper.readTree(response).path("status").textValue()).as(tool.getToolName()).isEqualTo("error");
        }
        assertThat(toolExecutors.stream()
                .map(ToolExecutor::getToolName)
                .filter(guardedToolNames::contains))
                .containsExactlyInAnyOrderElementsOf(guardedToolNames);
    }
}
