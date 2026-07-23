package com.stonewu.fusion.service.generation.video.consumer;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.infrastructure.queue.RedisTaskQueue;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategyRouter;
import com.stonewu.fusion.service.storage.MediaStorageService;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class VideoGenerationConsumerTests {

    @Test
        void submitTaskDefaultsWatermarkOffAndAudioOnWhenUnset() {
        RedisTaskQueue taskQueue = mock(RedisTaskQueue.class);
        VideoGenerationService videoGenerationService = mock(VideoGenerationService.class);
        AiModelService aiModelService = mock(AiModelService.class);
        GenerationModelCapabilityService capabilityService = mock(GenerationModelCapabilityService.class);
        VideoGenerationStrategyRouter strategyRouter = mock(VideoGenerationStrategyRouter.class);

        AiModel model = AiModel.builder()
                .id(101L)
                .status(1)
                .build();
        when(aiModelService.getById(101L)).thenReturn(model);
        when(videoGenerationService.create(any(VideoTask.class))).thenAnswer(invocation -> {
            VideoTask created = invocation.getArgument(0);
            created.setId(201L);
            return created;
        });

        VideoGenerationConsumer consumer = new VideoGenerationConsumer(
                taskQueue,
                videoGenerationService,
                aiModelService,
                capabilityService,
                strategyRouter,
                mock(MediaStorageService.class)
        );

        VideoTask task = VideoTask.builder()
                .modelId(101L)
                .prompt("test prompt")
                .build();

        consumer.submitTask(task);

        ArgumentCaptor<VideoTask> taskCaptor = ArgumentCaptor.forClass(VideoTask.class);
        verify(videoGenerationService).create(taskCaptor.capture());
        VideoTask createdTask = taskCaptor.getValue();
        assertThat(createdTask.getWatermark()).isFalse();
        assertThat(createdTask.getGenerateAudio()).isTrue();
    }

    @Test
        void submitTaskPreservesExplicitFlags() {
        RedisTaskQueue taskQueue = mock(RedisTaskQueue.class);
        VideoGenerationService videoGenerationService = mock(VideoGenerationService.class);
        AiModelService aiModelService = mock(AiModelService.class);
        GenerationModelCapabilityService capabilityService = mock(GenerationModelCapabilityService.class);
        VideoGenerationStrategyRouter strategyRouter = mock(VideoGenerationStrategyRouter.class);

        AiModel model = AiModel.builder()
                .id(102L)
                .status(1)
                .build();
        when(aiModelService.getById(102L)).thenReturn(model);
        when(videoGenerationService.create(any(VideoTask.class))).thenAnswer(invocation -> {
            VideoTask created = invocation.getArgument(0);
            created.setId(202L);
            return created;
        });

        VideoGenerationConsumer consumer = new VideoGenerationConsumer(
                taskQueue,
                videoGenerationService,
                aiModelService,
                capabilityService,
                strategyRouter,
                mock(MediaStorageService.class)
        );

        VideoTask task = VideoTask.builder()
                .modelId(102L)
                .prompt("test prompt")
                .watermark(true)
                .generateAudio(false)
                .build();

        consumer.submitTask(task);

        ArgumentCaptor<VideoTask> taskCaptor = ArgumentCaptor.forClass(VideoTask.class);
        verify(videoGenerationService).create(taskCaptor.capture());
        VideoTask createdTask = taskCaptor.getValue();
                assertThat(createdTask.getWatermark()).isTrue();
                assertThat(createdTask.getGenerateAudio()).isFalse();
    }
}