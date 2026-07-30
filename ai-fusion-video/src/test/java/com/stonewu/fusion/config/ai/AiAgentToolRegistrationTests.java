package com.stonewu.fusion.config.ai;

import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.ToolPermissionRisk;
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
}
