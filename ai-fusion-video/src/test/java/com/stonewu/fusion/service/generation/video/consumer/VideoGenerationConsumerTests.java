package com.stonewu.fusion.service.generation.video.consumer;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.infrastructure.queue.RedisTaskQueue;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.ReferenceImageTransportService;
import com.stonewu.fusion.service.generation.video.VideoFrameExtractor;
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
    void persistVideoItemsExtractsMissingFramesAndUsesFirstFrameAsCover() {
        VideoGenerationService videoGenerationService = mock(VideoGenerationService.class);
        MediaStorageService mediaStorageService = mock(MediaStorageService.class);
        VideoFrameExtractor videoFrameExtractor = mock(VideoFrameExtractor.class);
        VideoTask task = VideoTask.builder().id(201L).build();
        VideoItem item = VideoItem.builder()
                .id(301L)
                .taskId(201L)
                .videoUrl("https://example.com/generated.mp4")
                .build();

        when(videoGenerationService.listItems(201L)).thenReturn(List.of(item));
        when(mediaStorageService.downloadAndStore(item.getVideoUrl(), "videos"))
                .thenReturn("/media/videos/generated.mp4");
        when(videoFrameExtractor.extract("/media/videos/generated.mp4", true, true))
                .thenReturn(new VideoFrameExtractor.ExtractedFrames(
                        "/media/images/video-frames/first.jpg",
                        "/media/images/video-frames/last.jpg"));

        VideoGenerationConsumer consumer = new VideoGenerationConsumer(
                mock(RedisTaskQueue.class),
                videoGenerationService,
                mock(AiModelService.class),
                mock(ApiConfigService.class),
                mock(GenerationModelCapabilityService.class),
                mock(ReferenceImageTransportService.class),
                mock(VideoGenerationStrategyRouter.class),
                mediaStorageService,
                videoFrameExtractor
        );

        consumer.persistVideoItems(task);

        assertThat(item.getVideoUrl()).isEqualTo("/media/videos/generated.mp4");
        assertThat(item.getFirstFrameUrl()).isEqualTo("/media/images/video-frames/first.jpg");
        assertThat(item.getLastFrameUrl()).isEqualTo("/media/images/video-frames/last.jpg");
        assertThat(item.getCoverUrl()).isEqualTo(item.getFirstFrameUrl());
        verify(videoGenerationService).updateItem(item);
    }

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
                mock(ApiConfigService.class),
                capabilityService,
                mock(ReferenceImageTransportService.class),
                strategyRouter,
                mock(MediaStorageService.class),
                mock(VideoFrameExtractor.class)
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
                mock(ApiConfigService.class),
                capabilityService,
                mock(ReferenceImageTransportService.class),
                strategyRouter,
                mock(MediaStorageService.class),
                mock(VideoFrameExtractor.class)
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
