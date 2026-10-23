package com.stonewu.fusion.service.ai.tool;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.ToolPermissionRisk;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.tool.storyboard.BatchUpdateStoryboardToolExecutor;
import com.stonewu.fusion.service.ai.tool.storyboard.DeleteStoryboardChildResourceToolExecutor;
import com.stonewu.fusion.service.ai.tool.storyboard.ManageStoryboardChildrenToolExecutor;
import com.stonewu.fusion.service.ai.tool.storyboard.SaveStoryboardEpisodeToolExecutor;
import com.stonewu.fusion.service.ai.tool.storyboard.SaveStoryboardSceneShotsToolExecutor;
import com.stonewu.fusion.service.ai.tool.storyboard.StoryboardItemInsertToolExecutor;
import com.stonewu.fusion.service.ai.tool.storyboard.UpdateStoryboardItemFrameToolExecutor;
import com.stonewu.fusion.service.ai.tool.storyboard.UpdateStoryboardItemToolExecutor;
import com.stonewu.fusion.service.ai.tool.storyboard.UpdateStoryboardItemVideoToolExecutor;
import com.stonewu.fusion.service.ai.tool.storyboard.UpdateStoryboardSceneToolExecutor;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Map;
import static org.assertj.core.api.Assertions.assertThat;

class StoryboardToolSchemaTests {
    @Test
    void versionedToolsExposeSchemasAcceptedByTheRuntimeAndNativeArrayFields() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        List<ToolExecutor> tools = List.of(
                new SaveStoryboardEpisodeToolExecutor(null), new SaveStoryboardSceneShotsToolExecutor(null),
                new StoryboardItemInsertToolExecutor(null), new UpdateStoryboardItemToolExecutor(null),
                new UpdateStoryboardSceneToolExecutor(null), new UpdateStoryboardItemFrameToolExecutor(null),
                new UpdateStoryboardItemVideoToolExecutor(null), new DeleteStoryboardChildResourceToolExecutor(null),
                new ManageStoryboardChildrenToolExecutor(null), new BatchUpdateStoryboardToolExecutor(null));
        for (ToolExecutor tool : tools) {
            var prepared = AgentScopeToolSchema.prepare(mapper, tool.getParametersSchema(), tool.getToolName());
            assertThat(prepared.canonicalJson()).as(tool.getToolName()).isNotBlank();
        }
        var schema = mapper.readTree(new UpdateStoryboardItemToolExecutor(null).getParametersSchema());
        assertThat(schema.path("properties").path("characterIds").path("type").textValue()).isEqualTo("array");
        assertThat(schema.path("required").toString()).contains("expectedVersion");
        assertThat(schema.path("properties").has("sortOrder")).isFalse();
    }

    @Test
    void destructiveCollectionOperationsKeepTheirApprovalSensitivity() {
        var tool = new ManageStoryboardChildrenToolExecutor(null);
        assertThat(tool.mayRequireHighRiskApproval()).isTrue();
        for (String mode : List.of("REPLACE", "CLEAR", "DELETE")) {
            assertThat(tool.getPermissionRisk(Map.of("mode", mode))).isEqualTo(ToolPermissionRisk.HIGH_RISK);
        }
        for (String mode : List.of("APPEND", "REORDER")) {
            assertThat(tool.getPermissionRisk(Map.of("mode", mode))).isEqualTo(ToolPermissionRisk.EDIT);
        }
    }
}
