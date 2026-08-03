package com.stonewu.fusion.service.ai.tool;

import com.stonewu.fusion.service.ai.tool.storyboard.StoryboardQueryToolExecutor;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class StoryboardQueryToolExecutorTests {

    @Test
    void ordersItemsByEpisodeSceneAndShot() {
        StoryboardService storyboardService = mock(StoryboardService.class);
        ToolResourceAccessGuard accessGuard = mock(ToolResourceAccessGuard.class);
        StoryboardQueryToolExecutor executor = new StoryboardQueryToolExecutor(storyboardService, accessGuard);

        Storyboard storyboard = Storyboard.builder().id(5L).title("测试分镜").build();
        StoryboardEpisode episodeOne = StoryboardEpisode.builder()
                .id(11L)
                .storyboardId(5L)
                .episodeNumber(1)
                .sortOrder(0)
                .build();
        StoryboardEpisode episodeTwo = StoryboardEpisode.builder()
                .id(12L)
                .storyboardId(5L)
                .episodeNumber(2)
                .sortOrder(1)
                .build();
        StoryboardScene firstScene = StoryboardScene.builder()
                .id(101L)
                .episodeId(11L)
                .storyboardId(5L)
                .sceneNumber("1-1")
                .sortOrder(0)
                .build();
        StoryboardScene secondScene = StoryboardScene.builder()
                .id(102L)
                .episodeId(11L)
                .storyboardId(5L)
                .sceneNumber("1-2")
                .sortOrder(1)
                .build();
        StoryboardScene nextEpisodeScene = StoryboardScene.builder()
                .id(201L)
                .episodeId(12L)
                .storyboardId(5L)
                .sceneNumber("2-1")
                .sortOrder(0)
                .build();

        StoryboardItem firstShot = item(1001L, 11L, 101L, 0, "1");
        StoryboardItem secondShot = item(1002L, 11L, 101L, 1, "2");
        StoryboardItem nextSceneFirstShot = item(1003L, 11L, 102L, 0, "1");
        StoryboardItem nextEpisodeFirstShot = item(1004L, 12L, 201L, 0, "1");

        when(accessGuard.requireStoryboard(5L, 7L)).thenReturn(storyboard);
        when(storyboardService.listEpisodes(5L)).thenReturn(List.of(episodeOne, episodeTwo));
        when(storyboardService.listScenesByStoryboard(5L)).thenReturn(List.of(
                nextEpisodeScene,
                secondScene,
                firstScene));
        when(storyboardService.listItems(5L)).thenReturn(List.of(
                nextEpisodeFirstShot,
                nextSceneFirstShot,
                secondShot,
                firstShot));

        String result = executor.execute(
                "{\"storyboardId\":5}",
                ToolExecutionContext.builder().userId(7L).build());

        JSONArray items = JSONUtil.parseObj(result).getJSONArray("items");
        assertThat(items.getJSONObject(0).getLong("id")).isEqualTo(1001L);
        assertThat(items.getJSONObject(1).getLong("id")).isEqualTo(1002L);
        assertThat(items.getJSONObject(2).getLong("id")).isEqualTo(1003L);
        assertThat(items.getJSONObject(3).getLong("id")).isEqualTo(1004L);
        assertThat(items.getJSONObject(0).getStr("sceneNumber")).isEqualTo("1-1");
        assertThat(items.getJSONObject(0).getStr("shotNumber")).isEqualTo("1");
        assertThat(items.getJSONObject(2).getStr("sceneNumber")).isEqualTo("1-2");
        assertThat(items.getJSONObject(3).getStr("sceneNumber")).isEqualTo("2-1");
    }

    private static StoryboardItem item(
            Long id,
            Long episodeId,
            Long sceneId,
            Integer sortOrder,
            String shotNumber) {
        return StoryboardItem.builder()
                .id(id)
                .storyboardId(5L)
                .storyboardEpisodeId(episodeId)
                .storyboardSceneId(sceneId)
                .sortOrder(sortOrder)
                .shotNumber(shotNumber)
                .build();
    }
}
