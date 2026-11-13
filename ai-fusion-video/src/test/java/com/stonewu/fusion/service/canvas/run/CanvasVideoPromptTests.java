package com.stonewu.fusion.service.canvas.run;

import cn.hutool.json.JSONObject;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasGraphValidator;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasGraphValidation;
import com.stonewu.fusion.service.canvas.model.CanvasEdge;
import com.stonewu.fusion.service.canvas.model.CanvasEdgeKind;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasResourceReference;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.generation.job.GenerationSubmission;
import com.stonewu.fusion.service.generation.job.GenerationSubmissionService;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.generation.video.strategy.google.GoogleVideoProtocol;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import com.stonewu.fusion.service.resource.model.ResourceSnapshot;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class CanvasVideoPromptTests {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private final CanvasJson json = new CanvasJson(MAPPER);
    private final CanvasActor actor = new CanvasActor(42L, null);
    private final ResourceQueryService resources = mock(ResourceQueryService.class);
    private final CanvasRunSources sources = new CanvasRunSources(null, null, null, null, resources, null, json);
    private final CanvasRunInputs inputs = new CanvasRunInputs(new CanvasCapabilityRegistry(), sources);

    // 覆盖空视频提示词的环境镜头和对白镜头，验证镜头内容一直到供应商请求仍然完整。
    @ParameterizedTest
    @MethodSource("shots")
    void storyboardFieldsReachProviderRequestWithoutDedicatedVideoPrompt(ObjectNode data) {
        Ports shot = freezeShot(data);
        var asset = new Value("asset", CanvasValueType.ASSET_ITEM, null, null, null,
                new CanvasResourceReference(CanvasValueType.ASSET_ITEM, 2L, 1L),
                MAPPER.createObjectNode().put("name", "雨中山门").put("aiPrompt", "场景概念图，纯白背景"), null);
        var plan = videoPlan();
        Ports resolved = inputs.configured(plan,
                Map.of("prompt", shot.at("prompt"), "context", List.of(asset)), actor);
        VideoTask task = submit(plan, resolved);
        var body = GoogleVideoProtocol.request(task, model(), new JSONObject(), false);
        String sent = body.getJSONArray("input").getJSONObject(0).getJSONArray("content")
                .getJSONObject(0).getStr("text");

        assertThat(sent).contains("画面内容：" + data.path("content").asText(),
                "对白与旁白：" + data.path("dialogue").asText(), "景别：" + data.path("shotType").asText(),
                "名称：雨中山门").doesNotContain("场景概念图", "纯白背景", "videoPrompt", "sortOrder");
        if (data.hasNonNull("cameraMovement")) {
            assertThat(sent).contains("运镜：缓慢横移", "声音：雨声、远处低声喝骂", "字幕显出");
        }
        assertThat(task.getDuration()).isEqualTo(14);
        assertThat(task.getPrompt()).isEqualTo(sent);
        assertThat(data.path("videoPrompt").isNull()).isTrue();
    }

    @Test
    void dedicatedVideoInstructionsAreKeptAlongsideStoryboardDetails() {
        var data = MAPPER.createObjectNode().put("videoPrompt", "一镜到底，人物始终在画面左侧")
                .put("content", "守门人摆手让考生离开").put("dialogue", "请回吧。")
                .put("cameraMovement", "固定").put("soundEffect", "衣袖摩擦声");

        String prompt = freezeShot(data).at("prompt").getFirst().text();

        assertThat(prompt).contains("视频要求：一镜到底，人物始终在画面左侧", "画面内容：守门人摆手让考生离开",
                "对白与旁白：请回吧。", "运镜：固定", "音效：衣袖摩擦声");
    }

    // 已连接的空镜头或空文本不能由资产资料填充后继续提交。
    @ParameterizedTest
    @ValueSource(strings = {"", "   "})
    void connectedEmptyPromptIsRejectedBeforeAssetContextIsComposed(String prompt) {
        assertThatThrownBy(() -> inputs.configured(videoPlan(),
                Map.of("prompt", List.of(sources.text(prompt)), "context", List.of(sources.text("雨中山门"))), actor))
                .isInstanceOf(CanvasRunException.class).hasMessageContaining("连接的创作要求为空");
    }

    @Test
    void metadataOnlyShotDoesNotBecomeCreativeContent() {
        var data = MAPPER.createObjectNode().put("id", 1).put("duration", 10).put("sortOrder", 2)
                .put("status", 0).putNull("videoPrompt");

        assertThat(freezeShot(data).at("prompt").getFirst().text()).isEmpty();
    }

    @Test
    void planReportsEmptyShotBeforeAcceptingVideoExecution() {
        var document = mock(CanvasDocument.class);
        when(document.id()).thenReturn("canvas");
        when(document.projectId()).thenReturn(7L);
        var shot = new CanvasNode("shot", CanvasNodeKind.SHOT, 1, "镜头", null, MAPPER.createObjectNode(),
                new CanvasResourceReference(CanvasValueType.SHOT, 1L, 0L), 0, 0);
        when(resources.get(CanvasValueType.SHOT, 1L, actor)).thenReturn(new ResourceSnapshot(
                new ResourceHeader(CanvasValueType.SHOT, 1L, 7L, 0, false),
                MAPPER.createObjectNode().putNull("videoPrompt").put("sortOrder", 1)));
        var context = new CanvasNode("context", CanvasNodeKind.TEXT, 1, "场景设定", null,
                MAPPER.createObjectNode().put("content", "雨中山门"), null, 0, 0);
        var graph = new CanvasGraph(1, 7L, List.of(shot, context, videoPlan().node()), List.of(
                new CanvasEdge("prompt-edge", CanvasEdgeKind.EXECUTION, "shot", "prompt", "video", "prompt", 0, null, 0),
                new CanvasEdge("context-edge", CanvasEdgeKind.EXECUTION, "context", "text", "video", "context", 0, null, 0)));
        var validator = mock(CanvasGraphValidator.class);
        when(validator.validate(graph)).thenReturn(new CanvasGraphValidation(List.of("shot", "context", "video")));
        var environment = mock(CanvasRunEnvironment.class);
        when(environment.freeze(any(), eq(document), eq(actor))).thenReturn(MAPPER.createObjectNode());
        var executors = mock(CanvasRunExecutors.class);
        when(executors.supports(any())).thenReturn(true);
        var planner = new CanvasRunPlanner(validator, new CanvasCapabilityRegistry(), sources, environment,
                null, executors, json, null, null, null, null);

        var plan = planner.build(document, graph, new PlanRequest(0L, Scope.ALL, List.of(), Map.of(),
                new Budget(10, 0, 60, 1, 10), new GenerationClientScope(actor.userId(), null)), actor);

        assertThat(plan.issues()).singleElement().satisfies(issue -> {
            assertThat(issue.nodeId()).isEqualTo("video");
            assertThat(issue.code()).isEqualTo("EMPTY_SOURCE");
            assertThat(issue.message()).contains("连接的创作要求为空");
        });
    }

    private Ports freezeShot(ObjectNode data) {
        var document = mock(CanvasDocument.class);
        when(document.projectId()).thenReturn(7L);
        var source = new CanvasNode("shot", CanvasNodeKind.SHOT, 1, "镜头", null, MAPPER.createObjectNode(),
                new CanvasResourceReference(CanvasValueType.SHOT, 1L, 0L), 0, 0);
        when(resources.get(CanvasValueType.SHOT, 1L, actor)).thenReturn(
                new ResourceSnapshot(new ResourceHeader(CanvasValueType.SHOT, 1L, 7L, 0, false), data));
        return sources.freeze(document, source, actor, Set.of("prompt"));
    }

    private NodePlan videoPlan() {
        var node = new CanvasNode("video", CanvasNodeKind.VIDEO_GENERATION, 1, "视频", null,
                MAPPER.createObjectNode().put("modelId", 26).put("duration", 14).put("count", 1), null, 0, 0);
        var environment = MAPPER.createObjectNode().put("apiHash", "api");
        environment.set("model", json.object(model()));
        return new NodePlan(node, "hash", "EXECUTE", null, List.of(), List.of(), Map.of(), Ports.empty(),
                environment, 0, 14);
    }

    private AiModel model() {
        return AiModel.builder().id(26L).code("gemini-omni-1.2-flash").config("{}").build();
    }

    private VideoTask submit(NodePlan plan, Ports resolved) {
        var submissions = mock(GenerationSubmissionService.class);
        var submission = mock(GenerationSubmission.class);
        when(submission.id()).thenReturn("submission");
        when(submissions.videoFrozen(any(), anyString(), any(), anyString(), any(), anyString()))
                .thenReturn(submission);
        var executor = new CanvasGenerationExecutor(submissions, null, null, null, json, null, null);
        var node = mock(NodeRun.class);
        when(node.id()).thenReturn("node-run");
        when(node.plan()).thenReturn(plan);
        when(node.inputs()).thenReturn(resolved);
        var run = mock(Run.class);
        var runPlan = mock(Plan.class);
        when(run.actor()).thenReturn(actor);
        when(run.canvasId()).thenReturn("canvas");
        when(run.plan()).thenReturn(runPlan);
        when(runPlan.graph()).thenReturn(new CanvasGraph(1, 7L, List.of(), List.of()));

        executor.step(run, node, null);

        var task = ArgumentCaptor.forClass(VideoTask.class);
        verify(submissions).videoFrozen(task.capture(), anyString(), any(), eq("node-run"), any(), eq("api"));
        return task.getValue();
    }

    private static Stream<ObjectNode> shots() {
        return Stream.of(
                MAPPER.createObjectNode().putNull("videoPrompt").put("shotType", "远景")
                        .put("content", "雨滴落入路边水洼，字幕显出「今日宜出行，不宜作弊」。")
                        .put("dialogue", "农历二月初二，宜出行。")
                        .put("cameraMovement", "缓慢横移").put("sound", "雨声、远处低声喝骂"),
                MAPPER.createObjectNode().putNull("videoPrompt").put("shotType", "中景")
                        .put("content", "戴不凡揭晓救人者就是自己，弟子愣住；他随即摆手劝退超龄考生。")
                        .put("dialogue", "戴不凡：那位高人就是我。\n戴不凡：十八岁，超了年龄，你回去吧。"));
    }
}
