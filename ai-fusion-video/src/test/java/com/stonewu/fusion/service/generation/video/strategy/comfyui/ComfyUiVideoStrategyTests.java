package com.stonewu.fusion.service.generation.video.strategy.comfyui;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiExecutionContext;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiGenerationExecutor;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiPreparedSubmission;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiStoredOutput;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiJobResult;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InOrder;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ComfyUiVideoStrategyTests {

    @Mock
    private AiModelService aiModelService;
    @Mock
    private VideoGenerationService videoGenerationService;
    @Mock
    private ComfyUiGenerationExecutor executor;

    private ComfyUiVideoStrategy strategy;
    private AiModel model;
    private VideoTask task;
    private ComfyUiExecutionContext context;

    @BeforeEach
    void setUp() {
        strategy = new ComfyUiVideoStrategy(aiModelService, videoGenerationService, executor);
        model = AiModel.builder().id(6L).modelType(3).config("{\"defaultFps\":24}").build();
        task = VideoTask.builder()
                .id(11L)
                .taskId("video-task")
                .modelId(6L)
                .workflowVersionId(21L)
                .prompt("test")
                .resolution("1280x720")
                .duration(5)
                .count(1)
                .generateAudio(true)
                .build();
        context = new ComfyUiExecutionContext(model, ApiConfig.builder().id(7L).build(), null, null);
        when(aiModelService.getById(6L)).thenReturn(model);
        when(executor.resolveContext(model, 21L)).thenReturn(context);
    }

    @Test
    void submitPersistsPlatformPromptIdBeforeRemoteSubmission() {
        String promptId = UUID.randomUUID().toString();
        VideoItem item = VideoItem.builder().id(32L).taskId(11L).build();
        ComfyUiPreparedSubmission submission = new ComfyUiPreparedSubmission(
                context, promptId, new ObjectMapper().createObjectNode());
        when(videoGenerationService.listItems(11L)).thenReturn(List.of(item));
        when(executor.prepare(eq(context), eq("video-task"), any(Map.class))).thenReturn(submission);

        String result = strategy.submit(task);

        assertThat(result).isEqualTo(promptId);
        assertThat(item.getPlatformTaskId()).isEqualTo(promptId);
        InOrder order = inOrder(videoGenerationService, executor);
        order.verify(videoGenerationService).updateItem(item);
        order.verify(executor).submit(submission);
    }

    @Test
    void pollMapsPrimaryVideoAndCoverOutputs() {
        String promptId = UUID.randomUUID().toString();
        VideoItem item = VideoItem.builder().id(32L).taskId(11L).build();
        ComfyUiJobResult job = new ComfyUiJobResult(
                promptId, "completed", new ObjectMapper().createObjectNode(), null, null);
        when(executor.waitForJob(eq(context), eq(promptId), anyLong(), anyLong())).thenReturn(job);
        when(executor.storeOutputs(context, job)).thenReturn(List.of(
                new ComfyUiStoredOutput("video", "primary", "/media/comfy/result.mp4", 456L),
                new ComfyUiStoredOutput("image", "cover", "/media/comfy/cover.png", 78L)));
        when(videoGenerationService.listItems(11L)).thenReturn(List.of(item));

        strategy.poll(promptId, task);

        assertThat(item.getVideoUrl()).isEqualTo("/media/comfy/result.mp4");
        assertThat(item.getCoverUrl()).isEqualTo("/media/comfy/cover.png");
        assertThat(item.getFileSize()).isEqualTo(456L);
        assertThat(item.getDuration()).isEqualTo(5);
        assertThat(item.getStatus()).isEqualTo(1);
        assertThat(task.getSuccessCount()).isEqualTo(1);
        verify(videoGenerationService).updateItem(item);
        verify(videoGenerationService).update(task);
    }
}
