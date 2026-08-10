package com.stonewu.fusion.service.ai.tool;

import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.asset.Asset;
import com.stonewu.fusion.entity.asset.AssetItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.tool.asset.DeleteAssetResourceToolExecutor;
import com.stonewu.fusion.service.ai.tool.script.DeleteScriptChildResourceToolExecutor;
import com.stonewu.fusion.service.ai.tool.storyboard.DeleteStoryboardChildResourceToolExecutor;
import com.stonewu.fusion.service.asset.AssetService;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.script.ScriptService;
import com.stonewu.fusion.service.storyboard.StoryboardService;
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
    void deletesAssetItemOnlyAfterCheckingItsParentAsset() {
        AssetService assets = mock(AssetService.class);
        DeleteAssetResourceToolExecutor executor = new DeleteAssetResourceToolExecutor(assets);
        AssetItem item = AssetItem.builder().id(17L).assetId(7L).build();
        Asset asset = Asset.builder().id(7L).build();
        when(assets.getItemById(17L)).thenReturn(item);
        when(assets.getById(7L)).thenReturn(asset);
        when(assets.canAccessAsset(asset, 9L)).thenReturn(true);

        String result = executor.execute(
                "{\"resourceType\":\"item\",\"resourceId\":17}",
                CONTEXT);

        assertThat(JSONUtil.parseObj(result).getStr("status")).isEqualTo("success");
        verify(assets).deleteItem(17L);
        verify(assets, never()).delete(any());
    }

    @Test
    void refusesToDeleteTheFixedScriptContainer() {
        ScriptService scripts = mock(ScriptService.class);
        ProjectService projects = mock(ProjectService.class);
        DeleteScriptChildResourceToolExecutor executor = new DeleteScriptChildResourceToolExecutor(scripts, projects);

        String result = executor.execute(
                "{\"resourceType\":\"script\",\"resourceId\":3}",
                CONTEXT);

        assertThat(JSONUtil.parseObj(result).getStr("status")).isEqualTo("error");
        verify(scripts, never()).deleteEpisode(any());
        verify(scripts, never()).deleteScene(any());
    }

    @Test
    void refusesToDeleteTheFixedStoryboardContainer() {
        StoryboardService storyboards = mock(StoryboardService.class);
        ProjectService projects = mock(ProjectService.class);
        DeleteStoryboardChildResourceToolExecutor executor =
                new DeleteStoryboardChildResourceToolExecutor(storyboards, projects);

        String result = executor.execute(
                "{\"resourceType\":\"storyboard\",\"resourceId\":5}",
                CONTEXT);

        assertThat(JSONUtil.parseObj(result).getStr("status")).isEqualTo("error");
        verify(storyboards, never()).deleteEpisode(any());
        verify(storyboards, never()).deleteScene(any());
        verify(storyboards, never()).deleteItem(any());
    }
}
