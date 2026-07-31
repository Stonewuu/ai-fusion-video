package com.stonewu.fusion.service.ai.tool;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.tool.asset.AssetCreateToolExecutor;
import com.stonewu.fusion.service.ai.tool.script.ManageScriptSceneItemsToolExecutor;
import com.stonewu.fusion.service.ai.tool.script.UpdateScriptSceneItemToolExecutor;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.asset.AssetService;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.script.ScriptService;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class AssetCreateToolExecutorTests {

    @Test
    void exposesAValidAgentScopeJsonSchema() {
        assertValid(new AssetCreateToolExecutor(
                mock(AssetService.class),
                mock(ProjectService.class)));
    }

    @Test
    void dialogueArraysExposeProviderCompatibleItemSchemas() {
        ScriptService scripts = mock(ScriptService.class);
        ToolResourceAccessGuard accessGuard = mock(ToolResourceAccessGuard.class);

        assertValid(new UpdateScriptSceneItemToolExecutor(scripts, accessGuard));
        assertValid(new ManageScriptSceneItemsToolExecutor(scripts, accessGuard));
    }

    private void assertValid(ToolExecutor executor) {
        AgentScopeToolSchema.PreparedSchema schema = AgentScopeToolSchema.prepare(
                new ObjectMapper(),
                executor.getParametersSchema(),
                executor.getToolName());

        assertThat(schema.canonicalJson()).isNotBlank();
    }
}
