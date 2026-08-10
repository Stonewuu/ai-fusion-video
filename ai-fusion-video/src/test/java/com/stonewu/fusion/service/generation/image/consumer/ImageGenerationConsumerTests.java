package com.stonewu.fusion.service.generation.image.consumer;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.infrastructure.queue.RedisTaskQueue;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowService;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.ReferenceImageTransportService;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategyRouter;
import com.stonewu.fusion.service.storage.MediaStorageService;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ImageGenerationConsumerTests {

    @Test
    void submitPinsCurrentlyPublishedWorkflowVersionBeforeQueueing() {
        RedisTaskQueue taskQueue = mock(RedisTaskQueue.class);
        ImageGenerationService generationService = mock(ImageGenerationService.class);
        AiModelService aiModelService = mock(AiModelService.class);
        ComfyUiWorkflowService workflowService = mock(ComfyUiWorkflowService.class);
        AiModel model = AiModel.builder()
                .id(5L)
                .status(1)
                .comfyuiWorkflowId(12L)
                .build();
        when(aiModelService.getById(5L)).thenReturn(model);
        when(workflowService.requireWorkflow(12L)).thenReturn(ComfyUiWorkflow.builder()
                .id(12L)
                .status(1)
                .activeVersionId(21L)
                .build());
        when(workflowService.requireVersion(21L)).thenReturn(
                ComfyUiWorkflowVersion.builder().id(21L).workflowId(12L).published(true).build());
        when(generationService.create(any(ImageTask.class)))
                .thenAnswer(invocation -> {
                    ImageTask created = invocation.getArgument(0);
                    created.setId(10L);
                    return created;
                });
        ImageGenerationConsumer consumer = new ImageGenerationConsumer(
                taskQueue,
                generationService,
                aiModelService,
                mock(ApiConfigService.class),
                mock(GenerationModelCapabilityService.class),
                mock(ReferenceImageTransportService.class),
                mock(ImageGenerationStrategyRouter.class),
                mock(MediaStorageService.class),
                workflowService);
        ImageTask task = ImageTask.builder().modelId(5L).prompt("test").count(1).build();

        consumer.submitTask(task);

        assertThat(task.getWorkflowVersionId()).isEqualTo(21L);
        verify(taskQueue).push(eq("image_generation:model:5"), any(String.class));
        consumer.shutdownWorkerExecutor();
    }

    @Test
    void cancelQueuedTaskRemovesExactQueueEntryAndMarksPendingItemsCancelled() {
        RedisTaskQueue taskQueue = mock(RedisTaskQueue.class);
        ImageGenerationService generationService = mock(ImageGenerationService.class);
        ImageTask task = ImageTask.builder()
                .id(10L)
                .taskId("task-1")
                .userId(9L)
                .modelId(5L)
                .status(0)
                .build();
        ImageItem item = ImageItem.builder().id(20L).taskId(10L).status(0).build();
        when(generationService.getByTaskId("task-1")).thenReturn(task);
        when(generationService.listItems(10L)).thenReturn(List.of(item));
        when(taskQueue.remove("image_generation:model:5", "task-1")).thenReturn(true);
        ImageGenerationConsumer consumer = new ImageGenerationConsumer(
                taskQueue,
                generationService,
                mock(AiModelService.class),
                mock(ApiConfigService.class),
                mock(GenerationModelCapabilityService.class),
                mock(ReferenceImageTransportService.class),
                mock(ImageGenerationStrategyRouter.class),
                mock(MediaStorageService.class),
                mock(ComfyUiWorkflowService.class));

        boolean cancelled = consumer.cancelTask("task-1", 9L);

        assertThat(cancelled).isTrue();
        assertThat(item.getStatus()).isEqualTo(2);
        assertThat(item.getErrorMsg()).isEqualTo("用户取消");
        verify(generationService).updateItem(item);
        verify(generationService).updateStatus(10L, 3, "用户取消");
        consumer.shutdownWorkerExecutor();
    }
}
