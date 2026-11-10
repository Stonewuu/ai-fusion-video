package com.stonewu.fusion.service.generation;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.tool.generation.GenerateImageToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeGeneration;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.job.GenerationSubmission;
import com.stonewu.fusion.service.generation.job.GenerationSubmissionService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import com.stonewu.fusion.service.resource.model.ResourceSnapshot;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** 验证实际生图请求沿用镜头关联，不依赖模型手写参考图数组。 */
class StoryboardFrameInputsTests {
    final ObjectMapper mapper = new ObjectMapper();
    final ResourceQueryService resources = mock(ResourceQueryService.class);
    final AgentRunMapper runs = mock(AgentRunMapper.class);
    final StoryboardFrameInputs inputs = new StoryboardFrameInputs(resources, runs);
    final CanvasActor actor = new CanvasActor(7, null);
    final ResourceCommandContext mutation = new ResourceCommandContext(actor, "agent", null, "frame-run", "image-call", "worker", 1L);
    final ToolExecutionContext invocation = ToolExecutionContext.forInvocation(7L, 1, 7L, mutation);
    final ObjectNode shot = mapper.createObjectNode().put("sceneAssetItemId", 51).put("firstFrameImageUrl", "/first.png");

    StoryboardFrameInputsTests() {
        var run = new AgentRun();
        run.setAgentType("storyboard_frame_executor");
        when(runs.selectByRunId("frame-run")).thenReturn(run);
        shot.putArray("characterIds").add(52);
        shot.putArray("propIds").add(53);
        snapshot(SHOT, 10, 4, shot);
        asset(51, "scene", "道路", "/road.png");
        asset(52, "character", "旅人", "/person.png");
        asset(53, "prop", "马车", "/cart.png");
    }

    @Test
    void firstAndLastFramesUseBoundSceneCharacterAndPropInStableOrder() {
        var first = inputs.prepare(10, 4, 2, "first", actor);
        assertThat(first.references()).extracting(StoryboardFrameInputs.Reference::imageUrl)
                .containsExactly("/road.png", "/person.png", "/cart.png");
        var last = inputs.prepare(10, 4, 2, "last", actor);
        assertThat(last.references()).extracting(StoryboardFrameInputs.Reference::imageUrl)
                .containsExactly("/first.png", "/road.png", "/person.png", "/cart.png");
        assertThat(last.prompt("旅人正在走向马车")).contains("图片2：场景「道路」", "图片3：人物「旅人」", "图片4：道具「马车」");
    }

    @Test
    void absentSceneAndMissingSceneImageStopBeforeSubmission() throws Exception {
        var submissions = mock(GenerationSubmissionService.class);
        var executor = executor(submissions);
        shot.remove("sceneAssetItemId");
        assertThat(mapper.readTree(executor.execute(request(), invocation)).path("status").asText()).isEqualTo("error");
        shot.put("sceneAssetItemId", 51);
        asset(51, "scene", "道路", "");
        assertThat(mapper.readTree(executor.execute(request(), invocation)).path("message").asText()).contains("场景", "参考图");
        verifyNoInteractions(submissions);
    }

    @Test
    void staleSceneVersionCannotSilentlyUseANewReference() {
        assertThatThrownBy(() -> inputs.prepare(10, 3, 2, "first", actor)).isInstanceOf(CanvasException.class);
        assertThatThrownBy(() -> inputs.prepare(10, 4, 1, "first", actor)).isInstanceOf(CanvasException.class);
    }

    @Test
    void existingGenerateImageSubmitsAutomaticReferencesAndReturnsTheExactPrompt() throws Exception {
        var submissions = mock(GenerationSubmissionService.class);
        var accepted = mock(GenerationSubmission.class);
        when(accepted.id()).thenReturn("submission");
        when(accepted.state()).thenReturn("SUBMITTING");
        when(submissions.image(any(), anyString(), eq(mutation), isNull())).thenReturn(accepted);
        var result = mapper.readTree(executor(submissions).execute(request(), invocation));
        assertThat(result.path("status").asText()).isEqualTo("pending");
        var request = ArgumentCaptor.forClass(ImageTask.class);
        verify(submissions).image(request.capture(), anyString(), eq(mutation), isNull());
        assertThat(mapper.readTree(request.getValue().getRefImageUrls())).isEqualTo(mapper.valueToTree(List.of("/road.png", "/person.png", "/cart.png")));
        assertThat(result.path("framePrompt").asText()).isEqualTo(request.getValue().getPrompt()).contains("图片1：场景「道路」");
        assertThat(result.path("expectedVersion").intValue()).isEqualTo(4);
    }

    private String request() {
        return """
                {"prompt":"旅人在道路上走向马车","storyboardItemId":10,"frameType":"first",
                 "expectedVersion":4,"expectedSceneAssetVersion":2}
                """;
    }

    private GenerateImageToolExecutor executor(GenerationSubmissionService submissions) {
        var models = mock(AiModelService.class);
        when(models.getDefaultByType(2)).thenReturn(AiModel.builder().id(15L).code("image-model").build());
        return new GenerateImageToolExecutor(models, mock(ImageGenerationService.class), submissions,
                mock(GenerationModelCapabilityService.class), new CanvasJson(mapper), mock(CanvasAgentNodeGeneration.class), inputs);
    }

    private void asset(long id, String type, String name, String url) {
        snapshot(ASSET_ITEM, id, 2, mapper.createObjectNode().put("assetId", id + 100).put("name", name).put("imageUrl", url));
        snapshot(ASSET, id + 100, 2, mapper.createObjectNode().put("type", type));
    }

    private void snapshot(CanvasValueType type, long id, int version, ObjectNode data) {
        when(resources.get(type, id, actor)).thenReturn(new ResourceSnapshot(new ResourceHeader(type, id, 4L, version, false), data));
    }
}
