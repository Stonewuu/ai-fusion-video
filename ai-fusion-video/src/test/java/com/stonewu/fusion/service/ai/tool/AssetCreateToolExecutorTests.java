package com.stonewu.fusion.service.ai.tool;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.tool.asset.AssetCreateToolExecutor;
import com.stonewu.fusion.service.ai.tool.script.ManageScriptSceneItemsToolExecutor;
import com.stonewu.fusion.service.ai.tool.script.UpdateScriptSceneItemToolExecutor;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.tool.asset.AssetToolCommands;
import com.stonewu.fusion.service.ai.tool.script.ScriptToolCommands;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class AssetCreateToolExecutorTests {

    @Test
    void exposesAValidAgentScopeJsonSchema() {
        assertValid(new AssetCreateToolExecutor(mock(AssetToolCommands.class)));
    }

    @Test
    void dialogueArraysExposeProviderCompatibleItemSchemas() {
        ScriptToolCommands commands = mock(ScriptToolCommands.class);

        assertValid(new UpdateScriptSceneItemToolExecutor(commands));
        assertValid(new ManageScriptSceneItemsToolExecutor(commands));
    }

    private void assertValid(ToolExecutor executor) {
        AgentScopeToolSchema.PreparedSchema schema = AgentScopeToolSchema.prepare(
                new ObjectMapper(),
                executor.getParametersSchema(),
                executor.getToolName());

        assertThat(schema.canonicalJson()).isNotBlank();
    }
}
