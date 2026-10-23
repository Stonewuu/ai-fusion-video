package com.stonewu.fusion.service.ai.tool;

import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.asset.Asset;
import com.stonewu.fusion.entity.asset.AssetItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.tool.asset.DeleteAssetResourceToolExecutor;
import com.stonewu.fusion.service.ai.tool.script.DeleteScriptChildResourceToolExecutor;
import com.stonewu.fusion.service.ai.tool.storyboard.DeleteStoryboardChildResourceToolExecutor;
import com.stonewu.fusion.service.ai.tool.asset.AssetToolCommands;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.ai.tool.script.ScriptToolCommands;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.fasterxml.jackson.databind.ObjectMapper;
import static org.mockito.Mockito.verifyNoInteractions;
import com.stonewu.fusion.service.ai.tool.storyboard.StoryboardToolCommands;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class LifecycleToolExecutorTests {

    private static final ToolExecutionContext CONTEXT = ToolExecutionContext.builder()
            .userId(9L)
            .build();

    @Test
    void assetDeletionRejectsUnversionedOrUntrustedCallsBeforeWriting() {
        ResourceCommandService commands = mock(ResourceCommandService.class);
        AssetToolCommands assets = new AssetToolCommands(commands, mock(ResourceQueryService.class), new CanvasJson(new ObjectMapper()));
        var executor = new DeleteAssetResourceToolExecutor(assets);
        assertThat(JSONUtil.parseObj(executor.execute("{\"resourceType\":\"item\",\"resourceId\":17}", CONTEXT)).getStr("status")).isEqualTo("error");
        verifyNoInteractions(commands);
    }

    @Test
    void refusesToDeleteTheFixedScriptContainer() {
        ResourceCommandService commands = mock(ResourceCommandService.class);
        ScriptToolCommands scripts = new ScriptToolCommands(commands, mock(ResourceQueryService.class), new CanvasJson(new ObjectMapper()));
        DeleteScriptChildResourceToolExecutor executor = new DeleteScriptChildResourceToolExecutor(scripts);

        String result = executor.execute(
                "{\"resourceType\":\"script\",\"resourceId\":3}",
                CONTEXT);

        assertThat(JSONUtil.parseObj(result).getStr("status")).isEqualTo("error");
        verifyNoInteractions(commands);
    }

    @Test
    void refusesToDeleteTheFixedStoryboardContainer() {
        ResourceCommandService commands = mock(ResourceCommandService.class);
        StoryboardToolCommands storyboards = new StoryboardToolCommands(commands, mock(ResourceQueryService.class), new CanvasJson(new ObjectMapper()));
        DeleteStoryboardChildResourceToolExecutor executor = new DeleteStoryboardChildResourceToolExecutor(storyboards);
        String result = executor.execute("{\"resourceType\":\"storyboard\",\"resourceId\":5}", CONTEXT);
        assertThat(JSONUtil.parseObj(result).getStr("status")).isEqualTo("error");
        verifyNoInteractions(commands);
    }
}
