package com.stonewu.fusion.service.canvas.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.collaboration.CanvasPinnedOutputs;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import com.stonewu.fusion.service.canvas.model.CanvasEdge;
import com.stonewu.fusion.service.canvas.model.CanvasEdgeKind;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeActivity;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeView;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Ports;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.State;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CanvasGenerationInputPreviewTests {
    private final ObjectMapper json = new ObjectMapper();
    private final CanvasDocumentRepository documents = mock(CanvasDocumentRepository.class);
    private final CanvasGraphRepository graphs = mock(CanvasGraphRepository.class);
    private final CanvasAccessPolicy access = mock(CanvasAccessPolicy.class);
    private final CanvasRunSources sources = mock(CanvasRunSources.class);
    private final CanvasRunService runs = mock(CanvasRunService.class);
    private final CanvasPinnedOutputs fixed = mock(CanvasPinnedOutputs.class);
    private final CanvasActor actor = new CanvasActor(7L, null);
    private final CanvasDocument document = mock(CanvasDocument.class);
    private final CanvasGenerationInputPreview preview = new CanvasGenerationInputPreview(documents, graphs,
            access, new CanvasCapabilityRegistry(), sources, fixed, runs, new CanvasValueExecutor(sources));

    @BeforeEach
    void prepare() {
        when(document.id()).thenReturn("canvas");
        when(document.revision()).thenReturn(4L);
        when(documents.require("canvas", false)).thenReturn(document);
        when(sources.text(anyString())).thenAnswer(call -> text(call.getArgument(0)));
    }

    // 预览需要保留图片连接顺序，并让文本拼接结果与节点执行使用同一规则。
    @Test
    void previewsConnectedTextAndImagesInInputOrderWithoutAnEarlierRun() {
        var first = node("first", CanvasNodeKind.TEXT);
        var second = node("second", CanvasNodeKind.TEXT);
        var combine = node("combine", CanvasNodeKind.TEXT_COMBINE);
        combine.config().put("separator", "，");
        var image = node("reference", CanvasNodeKind.IMAGE);
        var target = node("target", CanvasNodeKind.IMAGE_GENERATION);
        target.config().put("additionalPrompt", "另加柔光");
        var imageValue = new Value("image", CanvasValueType.IMAGE, null, "portrait", 1, null, null, null);
        freeze(first, "text", text("雨中的街道"));
        freeze(second, "text", text("撑伞的人"));
        freeze(image, "media", imageValue);
        graph(List.of(first, second, combine, image, target), List.of(
                edge("b", second, "text", combine, "texts", 1),
                edge("a", first, "text", combine, "texts", 0),
                edge("c", combine, "text", target, "prompt", 0),
                edge("d", image, "media", target, "referenceImages", 0)));

        var result = preview.read("canvas", "target", 4, actor);

        assertThat(result.issues()).isEmpty();
        assertThat(result.prompt()).isEqualTo("雨中的街道，撑伞的人");
        assertThat(result.inputs().get(1).values()).containsExactly(imageValue);
        assertThat(target.config().path("additionalPrompt").asText()).isEqualTo("另加柔光");
    }

    // 过期结果不可伪装成本次输入，用户需要先处理对应的生成节点。
    @Test
    void staleGeneratedImageIsReportedAndNotIncludedAsAReference() {
        var source = node("image", CanvasNodeKind.IMAGE_GENERATION);
        var target = node("video", CanvasNodeKind.VIDEO_GENERATION);
        graph(List.of(source, target), List.of(edge("link", source, "images", target, "referenceImages", 0)));
        var activity = mock(NodeActivity.class);
        var output = mock(NodeView.class);
        when(activity.node()).thenReturn(output);
        when(output.nodeId()).thenReturn(source.id());
        when(output.status()).thenReturn(State.SUCCEEDED);
        when(output.derivedState()).thenReturn("STALE");
        when(runs.nodeActivities("canvas", actor)).thenReturn(List.of(activity));

        var result = preview.read("canvas", target.id(), 4, actor);

        assertThat(result.inputs()).isEmpty();
        assertThat(result.issues()).singleElement().satisfies(issue -> {
            assertThat(issue.nodeId()).isEqualTo(source.id());
            assertThat(issue.message()).contains(source.title());
        });
    }

    @Test
    void sourcePermissionFailureDoesNotExposeItsContent() {
        var source = node("text", CanvasNodeKind.TEXT);
        var target = node("image", CanvasNodeKind.IMAGE_GENERATION);
        var value = text("不可公开的素材正文");
        freeze(source, "text", value);
        graph(List.of(source, target), List.of(edge("link", source, "text", target, "prompt", 0)));
        doThrow(new BusinessException(403, "没有访问此素材的权限")).when(sources).authorize(value, actor, true);

        var result = preview.read("canvas", target.id(), 4, actor);

        assertThat(result.inputs()).isEmpty();
        assertThat(result.prompt()).isEmpty();
        assertThat(result.issues()).hasSize(1);
    }

    // 文本面板的 AI 写作依赖同一预览：连线文字作为写作素材显示，写作要求不混入预览。
    @Test
    void previewsConnectedTextForTextGeneration() {
        var source = node("text", CanvasNodeKind.TEXT);
        var target = node("writer", CanvasNodeKind.TEXT_GENERATION);
        target.config().put("instruction", "扩写成场景描写");
        freeze(source, "text", text("雨后的小巷"));
        graph(List.of(source, target), List.of(edge("link", source, "text", target, "prompt", 0)));

        var result = preview.read("canvas", target.id(), 4, actor);

        assertThat(result.issues()).isEmpty();
        assertThat(result.prompt()).isEqualTo("雨后的小巷");
        assertThat(result.inputs()).singleElement().satisfies(input -> assertThat(input.nodeId()).isEqualTo(source.id()));
    }

    // 非生成节点没有可预览的生成输入，避免把常量节点当成生成任务处理。
    @Test
    void rejectsNodesWithoutQuickGeneration() {
        var source = node("text", CanvasNodeKind.TEXT);
        graph(List.of(source), List.of());

        assertThatThrownBy(() -> preview.read("canvas", source.id(), 4, actor))
                .hasMessageContaining("请选择生成文本、生图或生视频节点");
    }

    @Test
    void changedCanvasRevisionRejectsThePreview() {
        assertThatThrownBy(() -> preview.read("canvas", "target", 3, actor)).hasMessageContaining("连线已更新");
    }

    private CanvasNode node(String id, CanvasNodeKind kind) {
        return new CanvasNode(id, kind, 1, id, null, json.createObjectNode(), null, 0, 0);
    }

    private void freeze(CanvasNode node, String port, Value value) {
        when(sources.freeze(document, node, actor, Set.of(port))).thenReturn(new Ports(Map.of(port, List.of(value))));
    }

    private void graph(List<CanvasNode> nodes, List<CanvasEdge> edges) {
        when(graphs.load(document)).thenReturn(new CanvasGraph(1, null, nodes, edges));
    }

    private CanvasEdge edge(String id, CanvasNode source, String output, CanvasNode target, String input, int index) {
        return new CanvasEdge(id, CanvasEdgeKind.EXECUTION, source.id(), output, target.id(), input, index, null, 0);
    }

    private Value text(String content) {
        return new Value(content, CanvasValueType.TEXT, content, null, null, null, null, null);
    }
}
