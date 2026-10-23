package com.stonewu.fusion.service.generation.job;

import cn.hutool.json.JSONObject;
import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeGeneration;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowService;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategy;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategyRouter;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategyRouter;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.mapper.generation.RemoteGenerationJobMapper;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

/**
 * 保留旧 Consumer 的版本与参数覆盖，并针对明确选定的旧发布版本增加回归。
 */
class GenerationSubmissionServiceTests {
    final GenerationModelSnapshot models = mock(GenerationModelSnapshot.class);
    final ComfyUiWorkflowService workflows = mock(ComfyUiWorkflowService.class);
    final ImageGenerationService images = mock(ImageGenerationService.class);
    final VideoGenerationService videos = mock(VideoGenerationService.class);
    final ImageGenerationStrategyRouter imageProtocols = mock(ImageGenerationStrategyRouter.class);
    final VideoGenerationStrategyRouter videoProtocols = mock(VideoGenerationStrategyRouter.class);
    final ImageGenerationStrategy imageProtocol = mock(ImageGenerationStrategy.class);
    final VideoGenerationStrategy videoProtocol = mock(VideoGenerationStrategy.class);
    final GenerationModelCapabilityService capabilities = mock(GenerationModelCapabilityService.class);
    final ResourceCommandContext actor = ResourceCommandContext.user(new CanvasActor(9, null), null);
    final AiModel model = AiModel.builder().id(5L).apiConfigId(7L).comfyuiWorkflowId(12L).build();
    final GenerationSubmissionService service = new GenerationSubmissionService(mock(GenerationSubmissionMapper.class),
            mock(RemoteGenerationJobMapper.class), new CanvasJson(new ObjectMapper().findAndRegisterModules()),
            mock(GenerationExecutionContexts.class), models, capabilities, workflows,
            images, videos, mock(GenerationJobRepository.class), mock(GenerationOutbox.class),
            mock(GenerationRequestRepository.class), imageProtocols, videoProtocols,
            mock(GenerationInputVersions.class), mock(CanvasAgentNodeGeneration.class));

    @BeforeEach
    void configure() {
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(new MybatisConfiguration(), ""),
                GenerationSubmissionEntity.class);
        when(models.freeze(5L, GenerationKind.IMAGE)).thenReturn(model);
        when(models.freeze(eq(5L), eq(GenerationKind.IMAGE), any(Long.class))).thenReturn(model);
        when(models.freeze(5L, GenerationKind.VIDEO)).thenReturn(model);
        when(models.requireApi(7L)).thenReturn(ApiConfig.builder().id(7L).build());
        when(workflows.requireWorkflow(12L)).thenReturn(
                ComfyUiWorkflow.builder().id(12L).status(1).activeVersionId(21L).build());
        when(workflows.requireVersion(21L)).thenReturn(
                ComfyUiWorkflowVersion.builder().id(21L).workflowId(12L).published(true).build());
        when(images.create(any())).thenAnswer(call -> {
            ImageTask task = call.getArgument(0);
            task.setId(10L);
            return task;
        });
        when(images.createItem(any())).thenReturn(ImageItem.builder().id(30L).build());
        when(videos.create(any())).thenAnswer(call -> {
            VideoTask task = call.getArgument(0);
            task.setId(11L);
            return task;
        });
        when(videos.createItem(any())).thenReturn(VideoItem.builder().id(31L).build());
        when(imageProtocols.resolve(eq(model), any())).thenReturn(imageProtocol);
        when(videoProtocols.resolve(model)).thenReturn(videoProtocol);
        when(imageProtocol.batchSize(any(), eq(model), any())).thenReturn(1);
        when(videoProtocol.batchSize(any(), eq(model), any())).thenReturn(1);
        when(capabilities.getMergedModelConfig(model)).thenReturn(new JSONObject());
    }

    @Test
    void unspecifiedWorkflowUsesCurrentPublishedVersion() {
        var task = ImageTask.builder().modelId(5L).prompt("测试").count(1).build();
        service.image(task, UUID.randomUUID().toString(), actor, null);
        assertThat(task.getWorkflowVersionId()).isEqualTo(21L);
        verify(images).create(task);
    }

    @Test
    void explicitlyChosenPublishedVersionIsNotReplacedByCurrentActiveVersion() {
        when(workflows.requireVersion(20L)).thenReturn(
                ComfyUiWorkflowVersion.builder().id(20L).workflowId(12L).published(true).build());
        var task = ImageTask.builder().modelId(5L).prompt("测试").workflowVersionId(20L).count(1).build();
        service.image(task, UUID.randomUUID().toString(), actor, null);
        assertThat(task.getWorkflowVersionId()).isEqualTo(20L);
    }

    @Test
    void admissionUsesTheFrozenVersionEvenIfAnotherVersionIsPublishedDuringSubmission() {
        model.setGenerationConfigSnapshot(true);
        model.setConfig("{\"comfyuiWorkflowVersionId\":20}");
        when(workflows.requireVersion(20L)).thenReturn(
                ComfyUiWorkflowVersion.builder().id(20L).workflowId(12L).published(true).build());
        var task = ImageTask.builder().modelId(5L).prompt("测试").count(1).build();
        service.image(task, UUID.randomUUID().toString(), actor, null);
        assertThat(task.getWorkflowVersionId()).isEqualTo(20L);
    }

    @Test
    void rejectsVersionMismatchBeforeCreatingAnyBusinessTask() {
        model.setGenerationConfigSnapshot(true);
        model.setConfig("{\"comfyuiWorkflowVersionId\":20}");
        var task = ImageTask.builder().modelId(5L).prompt("测试").workflowVersionId(21L).count(1).build();
        assertThatThrownBy(() -> service.image(task, UUID.randomUUID().toString(), actor, null))
                .hasMessageContaining("版本与已冻结的模型能力不一致");
        verifyNoInteractions(images);
    }

    @Test
    void unpublishedVersionCannotCreateAnyBusinessTask() {
        when(workflows.requireVersion(20L)).thenReturn(
                ComfyUiWorkflowVersion.builder().id(20L).workflowId(12L).published(false).build());
        var task = ImageTask.builder().modelId(5L).prompt("测试").workflowVersionId(20L).count(1).build();
        assertThatThrownBy(() -> service.image(task, UUID.randomUUID().toString(), actor, null)).isInstanceOf(
                GenerationJobException.class);
        verifyNoInteractions(images);
    }

    @Test
    void videoDefaultsRemainExplicitAndDoNotOverrideChosenFlags() {
        var omitted = VideoTask.builder().modelId(5L).prompt("测试").count(1).build();
        service.video(omitted, UUID.randomUUID().toString(), actor, null);
        assertThat(omitted.getWatermark()).isFalse();
        assertThat(omitted.getGenerateAudio()).isTrue();
        var explicit = VideoTask.builder().modelId(5L).prompt("测试").count(1).watermark(true).generateAudio(false)
                .build();
        service.video(explicit, UUID.randomUUID().toString(), actor, null);
        assertThat(explicit.getWatermark()).isTrue();
        assertThat(explicit.getGenerateAudio()).isFalse();
    }

    @Test
    void modelWithoutAudioSwitchKeepsTheOmittedOptionUnset() {
        when(capabilities.getMergedModelConfig(model)).thenReturn(new JSONObject().set("supportGenerateAudio", false));
        var task = VideoTask.builder().modelId(5L).prompt("雨夜灯塔").count(1).build();
        service.video(task, UUID.randomUUID().toString(), actor, null);
        assertThat(task.getGenerateAudio()).isNull();
        verify(videos).create(task);
    }
}
