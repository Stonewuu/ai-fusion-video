package com.stonewu.fusion.service.ai.tool;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.tool.storyboard.StoryboardToolCommands;
import com.stonewu.fusion.service.ai.tool.storyboard.UpdateStoryboardItemFrameToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class StoryboardFramePromptUpdateTests {
    private final ObjectMapper mapper = new ObjectMapper();
    private final ResourceCommandService commands = mock(ResourceCommandService.class);
    private final ResourceQueryService resources = mock(ResourceQueryService.class);
    private final UpdateStoryboardItemFrameToolExecutor executor = new UpdateStoryboardItemFrameToolExecutor(
            new StoryboardToolCommands(commands, resources, new CanvasJson(mapper)));
    private final ToolExecutionContext context = ToolExecutionContext.forInvocation(7L, 2, 81L,
            new ResourceCommandContext(new CanvasActor(7, 81L), "agent", null, "run", "call", "owner", 1L));

    @ParameterizedTest
    @ValueSource(strings = {"first", "last"})
    void promptRevisionKeepsExistingImageAndOtherFrameUntouched(String frame) throws Exception {
        when(commands.execute(any(), any())).thenReturn(new ResourceCommandReceipt("command", "request",
                mapper.createObjectNode().put("id", 42).put("version", 5), List.of()));
        var input = mapper.createObjectNode().put("storyboardItemId", 42).put("frameType", frame)
                .put("expectedVersion", 4).put("framePrompt", "手刚触到帘边，车帘仅开窄缝，人物仍在车内。");
        var result = mapper.readTree(executor.execute(input.toString(), context));
        assertThat(result.path("status").asText()).isEqualTo("success");
        var captured = ArgumentCaptor.forClass(ResourcePatchCommand.class);
        verify(commands).execute(captured.capture(), any());
        assertThat(captured.getValue().expectedVersion()).isEqualTo(4);
        assertThat(captured.getValue().patch().size()).isEqualTo(1);
        assertThat(captured.getValue().patch().path(frame + "FramePrompt").asText()).isEqualTo(input.path("framePrompt").asText());
        verifyNoInteractions(resources);
    }

    @Test
    void generationStillSavesImageAndExactFinalPromptTogether() throws Exception {
        when(commands.execute(any(), any())).thenReturn(new ResourceCommandReceipt("command", "request",
                mapper.createObjectNode().put("id", 42).put("version", 5), List.of()));
        executor.execute("""
                {"storyboardItemId":42,"frameType":"first","expectedVersion":4,
                 "imageUrl":"media://candidate","framePrompt":"俯视脚部，鞋底仍被泥浆包住。"}
                """, context);
        var captured = ArgumentCaptor.forClass(ResourcePatchCommand.class);
        verify(commands).execute(captured.capture(), any());
        assertThat(captured.getValue().patch().size()).isEqualTo(2);
        assertThat(captured.getValue().patch().path("firstFrameImageUrl").asText()).isEqualTo("media://candidate");
        assertThat(captured.getValue().patch().path("firstFramePrompt").asText()).isEqualTo("俯视脚部，鞋底仍被泥浆包住。");
    }

    @ParameterizedTest
    @ValueSource(strings = {"", ",\"framePrompt\":\" \"", ",\"framePrompt\":null", ",\"imageUrl\":null", ",\"imageUrl\":\"\""})
    void omittedContentAndAccidentalImageClearsAreRejected(String fields) throws Exception {
        var result = mapper.readTree(executor.execute(
                "{\"storyboardItemId\":42,\"frameType\":\"first\",\"expectedVersion\":4" + fields + "}", context));
        assertThat(result.path("status").asText()).isEqualTo("error");
        verifyNoInteractions(commands, resources);
    }

    @Test
    void versionConflictIsReportedWithoutRefreshingOrOverwriting() throws Exception {
        when(commands.execute(any(), any())).thenThrow(new CanvasException(CanvasErrorCode.VERSION_CONFLICT, "镜头已更新，请重新查看"));
        var result = mapper.readTree(executor.execute("""
                {"storyboardItemId":42,"frameType":"first","expectedVersion":4,"framePrompt":"修订后的首帧"}
                """, context));
        assertThat(result.path("errorCode").asText()).isEqualTo("VERSION_CONFLICT");
        verify(commands, times(1)).execute(any(), any());
        verifyNoInteractions(resources);
    }
}
