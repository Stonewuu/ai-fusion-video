package com.stonewu.fusion.service.generation.image.strategy.comfyui;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiExecutionContext;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiGenerationExecutor;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiPreparedSubmission;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiStoredOutput;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiJobResult;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
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
class ComfyUiImageStrategyTests {

    @Mock
    private AiModelService aiModelService;
    @Mock
    private ImageGenerationService imageGenerationService;
    @Mock
    private ComfyUiGenerationExecutor executor;

    private ComfyUiImageStrategy strategy;
    private AiModel model;
    private ImageTask task;
    private ComfyUiExecutionContext context;

    @BeforeEach
    void setUp() {
        strategy = new ComfyUiImageStrategy(aiModelService, imageGenerationService, executor);
        model = AiModel.builder().id(5L).modelType(2).build();
        task = ImageTask.builder()
                .id(10L)
                .taskId("image-task")
                .modelId(5L)
                .workflowVersionId(20L)
                .prompt("test")
                .width(512)
                .height(512)
                .count(1)
                .build();
        context = new ComfyUiExecutionContext(model, ApiConfig.builder().id(7L).build(), null, null);
        when(aiModelService.getById(5L)).thenReturn(model);
        when(executor.resolveContext(model, 20L)).thenReturn(context);
    }

    @Test
    void submitPersistsPlatformPromptIdBeforeRemoteSubmission() {
        String promptId = UUID.randomUUID().toString();
        ImageItem item = ImageItem.builder().id(31L).taskId(10L).build();
        ComfyUiPreparedSubmission submission = new ComfyUiPreparedSubmission(
                context, promptId, new ObjectMapper().createObjectNode());
        when(imageGenerationService.listItems(10L)).thenReturn(List.of(item));
        when(executor.prepare(eq(context), eq("image-task"), any(Map.class))).thenReturn(submission);

        String result = strategy.submit(task, context.apiConfig());

        assertThat(result).isEqualTo(promptId);
        assertThat(item.getPlatformTaskId()).isEqualTo(promptId);
        InOrder order = inOrder(imageGenerationService, executor);
        order.verify(imageGenerationService).updateItem(item);
        order.verify(executor).submit(submission);
    }

    @Test
    void pollWritesAlreadyPersistedComfyUiOutputWithoutSecondaryDownload() {
        String promptId = UUID.randomUUID().toString();
        ImageItem item = ImageItem.builder().id(31L).taskId(10L).build();
        ComfyUiJobResult job = new ComfyUiJobResult(
                promptId, "completed", new ObjectMapper().createObjectNode(), null, null);
        when(executor.waitForJob(eq(context), eq(promptId), anyLong(), anyLong())).thenReturn(job);
        when(executor.storeOutputs(context, job)).thenReturn(List.of(
                new ComfyUiStoredOutput("image", "primary", "/media/comfy/result.png", 123L)));
        when(imageGenerationService.listItems(10L)).thenReturn(List.of(item));

        strategy.poll(promptId, task, context.apiConfig());

        assertThat(item.getImageUrl()).isEqualTo("/media/comfy/result.png");
        assertThat(item.getFileSize()).isEqualTo(123L);
        assertThat(item.getStatus()).isEqualTo(1);
        assertThat(task.getSuccessCount()).isEqualTo(1);
        verify(imageGenerationService).updateItem(item);
        verify(imageGenerationService).update(task);
    }
}
