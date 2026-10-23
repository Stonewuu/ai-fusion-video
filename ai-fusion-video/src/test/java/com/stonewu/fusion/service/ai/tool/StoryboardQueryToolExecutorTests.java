package com.stonewu.fusion.service.ai.tool;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.tool.storyboard.StoryboardQueryToolExecutor;
import com.stonewu.fusion.service.ai.tool.storyboard.StoryboardToolQueries;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.StoryboardResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.system.SystemConfigService;
import org.junit.jupiter.api.Test;
import java.util.List;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

class StoryboardQueryToolExecutorTests {
    @Test
    void preservesObservedVersionsAndSceneMetadataWithAuthorizedMediaProjection() throws Exception {
        var boards = mock(StoryboardResourceQueryService.class);
        var configuration = mock(SystemConfigService.class);
        var mapper = new ObjectMapper();
        var actor = new CanvasActor(7, 81L);
        var root = mapper.createObjectNode().put("id", 5).put("version", 12).put("status", 1);
        var episode = root.putArray("episodes").addObject().put("id", 11).put("version", 8);
        episode.putArray("scenes").addObject().put("id", 101).put("version", 6).put("sceneNumber", "1-1").put("sceneHeading", "客厅");
        var item = mapper.createObjectNode().put("id", 1001).put("storyboardEpisodeId", 11).put("storyboardSceneId", 101)
                .put("version", 3).put("sceneVersion", 6).put("firstFrameImageUrl", "/media/image.png").put("generatedVideoUrl", "/media/private.mp4");
        when(boards.tree(CanvasValueType.STORYBOARD, 5, 2, actor)).thenReturn(root);
        when(boards.children(CanvasValueType.SHOT, CanvasValueType.STORYBOARD, 5, actor)).thenReturn(List.of(item));
        when(configuration.resolvePublicUrl("/media/image.png")).thenReturn("https://example.com/image.png");
        var executor = new StoryboardQueryToolExecutor(new StoryboardToolQueries(boards, mock(ResourceQueryService.class), configuration, new CanvasJson(mapper)));
        var invocation = ToolExecutionContext.forInvocation(7L, 2, 81L,
                new ResourceCommandContext(actor, "agent", null, "run", "call", "owner", 1L));
        var result = mapper.readTree(executor.execute("{\"storyboardId\":5}", invocation));
        assertThat(result.path("status").textValue()).isEqualTo("success");
        assertThat(result.path("resourceStatus").intValue()).isEqualTo(1);
        assertThat(result.path("version").intValue()).isEqualTo(12);
        var shot = result.path("items").get(0);
        assertThat(shot.path("version").intValue()).isEqualTo(3);
        assertThat(shot.path("sceneVersion").intValue()).isEqualTo(6);
        assertThat(shot.path("sceneNumber").textValue()).isEqualTo("1-1");
        assertThat(shot.path("firstFrameImageUrl").textValue()).isEqualTo("https://example.com/image.png");
        assertThat(shot.path("generatedVideoUrl").isNull()).isTrue();
        assertThat(shot.path("generatedVideoUrlStored").textValue()).isEqualTo("/media/private.mp4");
        verify(boards).tree(CanvasValueType.STORYBOARD, 5, 2, actor);
        verify(boards).children(CanvasValueType.SHOT, CanvasValueType.STORYBOARD, 5, actor);
    }
}
