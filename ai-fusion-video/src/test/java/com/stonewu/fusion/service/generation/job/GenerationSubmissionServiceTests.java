package com.stonewu.fusion.service.generation.job;

import cn.hutool.json.JSONObject;
import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import com.stonewu.fusion.entity.generation.RemoteGenerationJobEntity;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeGeneration;
import com.stonewu.fusion.service.ai.youkou.YoukouGenerationRequests;
import com.stonewu.fusion.service.generation.video.strategy.youkou.YoukouVideoStrategy;
import org.mockito.ArgumentCaptor;

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
import com.stonewu.fusion.service.media.ManagedStorageConfiguration;
import com.stonewu.fusion.service.media.model.MediaException;
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
 * 保护生成任务的工作流版本、参数保留和多结果拆分，避免重用参数或批量提交时丢失原始输入。
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
    final AssetImageReferencePolicy assetReferences = mock(AssetImageReferencePolicy.class);
    final ManagedStorageConfiguration storage = mock(ManagedStorageConfiguration.class);
    final ResourceCommandContext actor = ResourceCommandContext.user(new CanvasActor(9, null), null);
    final AiModel model = AiModel.builder().id(5L).apiConfigId(7L).comfyuiWorkflowId(12L).build();
    final RemoteGenerationJobMapper remoteJobs = mock(RemoteGenerationJobMapper.class);
    final GenerationSubmissionService service = new GenerationSubmissionService(mock(GenerationSubmissionMapper.class),
            remoteJobs, new CanvasJson(new ObjectMapper().findAndRegisterModules()),
            mock(GenerationExecutionContexts.class), models, capabilities, workflows,
            images, videos, mock(GenerationJobRepository.class), mock(GenerationOutbox.class),
            mock(GenerationRequestRepository.class), imageProtocols, videoProtocols,
            mock(GenerationInputVersions.class), mock(CanvasAgentNodeGeneration.class), assetReferences, storage);

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
    void invalidStorageRejectsImageBeforeCreatingAPaidGenerationTask() {
        when(storage.requireDefault()).thenThrow(MediaException.storageConfiguration("默认存储未启用"));
        var task = ImageTask.builder().modelId(5L).prompt("测试").count(1).build();
        assertThatThrownBy(() -> service.image(task, UUID.randomUUID().toString(), actor, null))
                .isInstanceOf(MediaException.class).hasMessage("默认存储未启用");
        verifyNoInteractions(images, remoteJobs, imageProtocols);
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

    @Test
    void youkouMultipleResultsCreateIndependentJobsWithTheSameReferences() throws Exception {
        model.setComfyuiWorkflowId(null);
        when(videoProtocols.resolve(model)).thenReturn(new YoukouVideoStrategy(mock(YoukouGenerationRequests.class)));
        when(videos.createItem(any())).thenReturn(VideoItem.builder().id(31L).build(),
                VideoItem.builder().id(32L).build(), VideoItem.builder().id(33L).build());
        var task = VideoTask.builder().modelId(5L).prompt("保留蓝色立方体").count(3).duration(4)
                .referenceVideoUrls("[\"media://22222222-2222-4222-8222-222222222222\"]").build();

        service.video(task, UUID.randomUUID().toString(), actor, null);

        var captured = ArgumentCaptor.forClass(RemoteGenerationJobEntity.class);
        verify(remoteJobs, times(3)).insertQueued(captured.capture());
        var queued = captured.getAllValues();
        assertThat(queued).extracting(RemoteGenerationJobEntity::getId).doesNotHaveDuplicates();
        assertThat(queued).extracting(RemoteGenerationJobEntity::getItemId).containsExactly(31L, 32L, 33L);
        assertThat(queued).extracting(RemoteGenerationJobEntity::getItemIndex).containsExactly(0, 1, 2);
        for (var job : queued) {
            var input = new ObjectMapper().readTree(job.getInputJson());
            assertThat(input.path("count").asInt()).isEqualTo(1);
            assertThat(input.path("prompt").asText()).isEqualTo("保留蓝色立方体");
            assertThat(input.path("referenceVideoUrls").asText()).isEqualTo(task.getReferenceVideoUrls());
        }
    }

    @Test
    void changedAssetReferencesCannotCreateAnotherPaidGeneration() {
        var agent = new ResourceCommandContext(actor.actor(), "agent", null, "asset-run", "retry", "worker", 1L);
        var task = ImageTask.builder().modelId(5L).prompt("去掉参考图后重试").count(1).build();
        doThrow(new GenerationJobException(409, "ASSET_IMAGE_REFERENCES_CHANGED", "参考图不能移除"))
                .when(assetReferences).enforce(task, agent);

        assertThatThrownBy(() -> service.image(task, UUID.randomUUID().toString(), agent, null))
                .isInstanceOf(GenerationJobException.class).hasMessageContaining("参考图不能移除");

        verifyNoInteractions(images, imageProtocols);
    }
}
