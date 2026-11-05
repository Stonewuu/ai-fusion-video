package com.stonewu.fusion.service.ai.tool;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.mapper.resource.ResourceReadMapper.ShotCount;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.tool.storyboard.StoryboardQueryToolExecutor;
import com.stonewu.fusion.service.ai.tool.storyboard.StoryboardToolQueries;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.StoryboardResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceSnapshot;
import com.stonewu.fusion.service.system.SystemConfigService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

class StoryboardQueryToolExecutorTests {
    private final StoryboardResourceQueryService boards = mock(StoryboardResourceQueryService.class);
    private final ResourceQueryService resources = mock(ResourceQueryService.class);
    private final SystemConfigService configuration = mock(SystemConfigService.class);
    private final ObjectMapper mapper = new ObjectMapper();
    private final CanvasActor actor = new CanvasActor(7, 81L);
    private final ObjectNode root = mapper.createObjectNode().put("id", 5).put("version", 12).put("status", 1);
    private final ObjectNode episode = mapper.createObjectNode().put("id", 11).put("storyboardId", 5).put("version", 8);
    private final StoryboardQueryToolExecutor executor = new StoryboardQueryToolExecutor(
            new StoryboardToolQueries(boards, resources, configuration, new CanvasJson(mapper)));
    private final ToolExecutionContext invocation = ToolExecutionContext.forInvocation(7L, 2, 81L,
            new ResourceCommandContext(actor, "agent", null, "run", "call", "owner", 1L));

    StoryboardQueryToolExecutorTests() {
        episode.putArray("scenes").addObject().put("id", 101).put("version", 6)
                .put("sceneNumber", "1-1").put("sceneHeading", "客厅");
    }

    @Test
    @DisplayName("整份分镜只查集与场次，2460 个镜头仍只返回数量，不读取镜头正文")
    void overviewReadsCountsAndNeverLoadsAllShots() throws Exception {
        root.putArray("episodes").add(episode);
        when(boards.tree(STORYBOARD, 5, 2, actor)).thenReturn(root);
        when(boards.shotCounts(5, actor)).thenReturn(List.of(new ShotCount(11L, 101L, 2460)));
        var result = mapper.readTree(executor.execute("{\"storyboardId\":5}", invocation));
        assertThat(result.path("status").asText()).isEqualTo("success");
        assertThat(result.path("scope").asText()).isEqualTo("overview");
        assertThat(result.path("resourceStatus").intValue()).isEqualTo(1);
        assertThat(result.path("version").intValue()).isEqualTo(12);
        assertThat(result.path("totalItems").intValue()).isEqualTo(2460);
        assertThat(result.path("episodes").get(0).path("itemCount").intValue()).isEqualTo(2460);
        assertThat(result.path("episodes").get(0).path("scenes").get(0).path("itemCount").intValue()).isEqualTo(2460);
        assertThat(result.has("items")).isFalse();
        assertThat(result.toString()).hasSizeLessThan(1000);
        verify(boards, never()).children(any(), any(), anyLong(), any());
        verifyNoInteractions(resources);
    }

    @Test
    @DisplayName("指定一集时保留镜头版本、场次信息和媒体访问结果")
    void episodeDetailsKeepVersionsAndAuthorizedMedia() throws Exception {
        when(resources.get(STORYBOARD, 5, actor)).thenReturn(new ResourceSnapshot(null, root));
        when(resources.get(STORYBOARD_EPISODE, 11, actor)).thenReturn(new ResourceSnapshot(null, episode));
        when(boards.tree(STORYBOARD_EPISODE, 11, 1, actor)).thenReturn(episode);
        var item = mapper.createObjectNode().put("id", 1001).put("storyboardEpisodeId", 11).put("storyboardSceneId", 101)
                .put("version", 3).put("sceneVersion", 6).put("firstFrameImageUrl", "/media/image.png")
                .put("generatedVideoUrl", "/media/private.mp4");
        when(boards.children(SHOT, STORYBOARD_EPISODE, 11, actor)).thenReturn(List.of(item));
        when(configuration.resolvePublicUrl("/media/image.png")).thenReturn("https://example.com/image.png");
        var result = mapper.readTree(executor.execute("{\"storyboardId\":5,\"storyboardEpisodeId\":11}", invocation));
        assertThat(result.path("status").asText()).isEqualTo("success");
        assertThat(result.path("scope").asText()).isEqualTo("episode");
        assertThat(result.path("episodes")).hasSize(1);
        var shot = result.path("items").get(0);
        assertThat(shot.path("version").intValue()).isEqualTo(3);
        assertThat(shot.path("sceneVersion").intValue()).isEqualTo(6);
        assertThat(shot.path("sceneNumber").textValue()).isEqualTo("1-1");
        assertThat(shot.path("firstFrameImageUrl").textValue()).isEqualTo("https://example.com/image.png");
        assertThat(shot.path("generatedVideoUrl").isNull()).isTrue();
        assertThat(shot.path("generatedVideoUrlStored").textValue()).isEqualTo("/media/private.mp4");
        verify(boards, never()).children(SHOT, STORYBOARD, 5, actor);
    }

    @Test
    @DisplayName("不能用其他分镜的集或集 ID 数组扩大镜头查询范围")
    void rejectsForeignEpisodesAndMultipleEpisodeRequests() throws Exception {
        when(resources.get(STORYBOARD, 5, actor)).thenReturn(new ResourceSnapshot(null, root));
        when(resources.get(STORYBOARD_EPISODE, 11, actor))
                .thenReturn(new ResourceSnapshot(null, episode.put("storyboardId", 9)));
        var foreign = mapper.readTree(executor.execute("{\"storyboardId\":5,\"storyboardEpisodeId\":11}", invocation));
        assertThat(foreign.path("status").asText()).isEqualTo("error");
        for (String request : List.of("{\"storyboardId\":5,\"storyboardEpisodeId\":[11,12]}",
                "{\"storyboardId\":5,\"storyboardEpisodeIds\":[11,12]}", "{\"storyboardId\":5,\"includeItems\":true}")) {
            assertThat(mapper.readTree(executor.execute(request, invocation)).path("status").asText()).isEqualTo("error");
        }
        verifyNoInteractions(boards);
    }
}
