package com.stonewu.fusion.service.generation.video.strategy.newapi;

import cn.hutool.json.JSONObject;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.model.AiModelMetadata;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class NewApiVideoStrategyTests {

    @Test
    void shouldPassMergedCapabilityPresetConfigToProtocolAdapter() {
        AiModelService aiModelService = mock(AiModelService.class);
        ApiConfigService apiConfigService = mock(ApiConfigService.class);
        VideoGenerationService videoGenerationService = mock(VideoGenerationService.class);
        AiModelMetadataResolver metadataResolver = mock(AiModelMetadataResolver.class);
        GenerationModelCapabilityService capabilityService = mock(GenerationModelCapabilityService.class);
        NewApiVideoProtocolRouter protocolRouter = mock(NewApiVideoProtocolRouter.class);
        NewApiVideoProtocolAdapter protocolAdapter = mock(NewApiVideoProtocolAdapter.class);

        AiModel model = AiModel.builder()
                .id(10L)
                .code("kling-v3")
                .modelProtocol("kling")
                .capabilityPresetCode("kling-v3-video")
                .apiConfigId(20L)
                .build();
        ApiConfig apiConfig = ApiConfig.builder()
                .id(20L)
                .platform("newapi")
                .apiUrl("https://example.com")
                .apiKey("test-key")
                .build();
        VideoTask task = VideoTask.builder()
                .id(30L)
                .taskId("video-task")
                .modelId(10L)
                .prompt("cinematic city")
                .build();
        VideoItem existingItem = VideoItem.builder()
                .platformTaskId("platform-task")
                .build();
        JSONObject mergedConfig = new JSONObject()
                .set("forwardGenerationFlagsViaMetadata", true)
                .set("frameRate", 30);
        AiModelMetadata metadata = new AiModelMetadata("newapi", "newapi", "kling");
        AtomicReference<NewApiVideoProtocolContext> capturedContext = new AtomicReference<>();

        when(aiModelService.getById(10L)).thenReturn(model);
        when(apiConfigService.getById(20L)).thenReturn(apiConfig);
        when(videoGenerationService.listItems(30L)).thenReturn(List.of(existingItem));
        when(capabilityService.getMergedModelConfig(model)).thenReturn(mergedConfig);
        when(metadataResolver.resolve(model, apiConfig)).thenReturn(metadata);
        when(protocolRouter.resolve(any())).thenAnswer(invocation -> {
            capturedContext.set(invocation.getArgument(0));
            return protocolAdapter;
        });

        NewApiVideoStrategy strategy = new NewApiVideoStrategy(
                aiModelService,
                apiConfigService,
                videoGenerationService,
                metadataResolver,
                capabilityService,
                protocolRouter
        );

        assertThat(strategy.submit(task)).isEqualTo("platform-task");
        assertThat(capturedContext.get()).isNotNull();
        assertThat((Object) capturedContext.get().modelConfig()).isSameAs(mergedConfig);
        verify(capabilityService).getMergedModelConfig(model);
    }
}
