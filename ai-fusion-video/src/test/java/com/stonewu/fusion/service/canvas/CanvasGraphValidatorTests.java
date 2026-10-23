package com.stonewu.fusion.service.canvas;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasConfigValidator;
import com.stonewu.fusion.service.canvas.model.CanvasEdge;
import com.stonewu.fusion.service.canvas.model.CanvasEdgeKind;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasNodeLayout;
import com.stonewu.fusion.service.canvas.model.CanvasResourceReference;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class CanvasGraphValidatorTests {
    private final ObjectMapper json = new ObjectMapper();
    private final CanvasGraphValidator validator = new CanvasGraphValidator(
            new CanvasCapabilityRegistry(), new CanvasConfigValidator());

    @Test
    void acceptsDraftNodesWithoutStartingOrRequiringModelExecution() {
        CanvasNode image = node(CanvasNodeKind.IMAGE_GENERATION, config());
        assertThat(validator.validate(graph(List.of(image), List.of())).topologicalNodeIds())
                .containsExactly(image.id());
    }

    @Test void everyRegisteredNodeCanBePlacedBeforeConfiguration() {
        var nodes = List.of(CanvasNodeKind.values()).stream().map(kind -> node(kind,config())).toList();
        assertThat(validator.validate(new CanvasGraph(1,1L,nodes,List.of())).topologicalNodeIds()).hasSize(nodes.size());
    }

    @Test void shotFirstAndLastFramesConnectToImageReferencesByValueType() {
        var shot = resourceNode(CanvasNodeKind.SHOT, CanvasValueType.SHOT);
        var image = node(CanvasNodeKind.IMAGE_GENERATION,config());
        var edges = List.of(edge(shot,"firstFrame",image,"referenceImages",0),edge(shot,"lastFrame",image,"referenceImages",1));
        assertThat(validator.validate(new CanvasGraph(1,1L,List.of(shot,image),edges)).topologicalNodeIds()).containsExactly(shot.id(),image.id());
        rejects(new CanvasGraph(1,1L,List.of(shot,image),List.of(edge(shot,"video",image,"referenceImages",0))),CanvasErrorCode.INCOMPATIBLE_PORT);
    }

    @Test
    void ordersImageCandidateSelectionBeforeVideoGeneration() {
        CanvasNode prompt = node(CanvasNodeKind.TEXT, config().put("content", "雨夜街道"));
        CanvasNode image = node(CanvasNodeKind.IMAGE_GENERATION, config().put("count", 4));
        CanvasNode selection = node(CanvasNodeKind.SELECTION, config().put("valueType", "image"));
        CanvasNode video = node(CanvasNodeKind.VIDEO_GENERATION, config());
        List<CanvasEdge> edges = List.of(edge(prompt, "text", image, "prompt", 0),
                edge(image, "images", selection, "candidates", 0), edge(selection, "selected", video, "firstFrame", 0));
        assertThat(validator.validate(graph(List.of(video, selection, image, prompt), edges)).topologicalNodeIds())
                .containsExactly(prompt.id(), image.id(), selection.id(), video.id());
    }

    @Test
    void rejectsImplicitlyTakingTheFirstImageFromMultipleCandidates() {
        CanvasNode image = node(CanvasNodeKind.IMAGE_GENERATION, config());
        CanvasNode video = node(CanvasNodeKind.VIDEO_GENERATION, config());
        rejects(graph(List.of(image, video), List.of(edge(image, "images", video, "firstFrame", 0))),
                CanvasErrorCode.INCOMPATIBLE_PORT);
    }

    @Test
    void rejectsVideoConnectedToAnImageInput() {
        CanvasNode source = node(CanvasNodeKind.VIDEO, config());
        CanvasNode target = node(CanvasNodeKind.VIDEO_GENERATION, config());
        rejects(graph(List.of(source, target), List.of(edge(source, "media", target, "firstFrame", 0))),
                CanvasErrorCode.INCOMPATIBLE_PORT);
    }

    @Test
    void rejectsTwoSourcesAtTheSameInputPosition() {
        CanvasNode first = node(CanvasNodeKind.IMAGE, config());
        CanvasNode second = node(CanvasNodeKind.IMAGE, config());
        CanvasNode video = node(CanvasNodeKind.VIDEO_GENERATION, config());
        rejects(graph(List.of(first, second, video), List.of(edge(first, "media", video, "referenceImages", 0),
                edge(second, "media", video, "referenceImages", 0))), CanvasErrorCode.DUPLICATE_INPUT);
    }

    @Test
    void preservesExplicitReferenceOrderingWithoutMixingItWithNodePosition() {
        CanvasNode first = node(CanvasNodeKind.IMAGE, config());
        CanvasNode second = node(CanvasNodeKind.IMAGE, config());
        CanvasNode video = node(CanvasNodeKind.VIDEO_GENERATION, config());
        CanvasGraph graph = graph(List.of(second, video, first), List.of(
                edge(first, "media", video, "referenceImages", 3), edge(second, "media", video, "referenceImages", 1)));
        validator.validate(graph);
        assertThat(graph.edges()).extracting(CanvasEdge::inputIndex).containsExactly(3, 1);
    }

    @Test
    void rejectsSimultaneousInlineAndConnectedPrompt() {
        CanvasNode text = node(CanvasNodeKind.TEXT, config());
        CanvasNode image = node(CanvasNodeKind.IMAGE_GENERATION, config().put("prompt", "手动提示词"));
        rejects(graph(List.of(text, image), List.of(edge(text, "text", image, "prompt", 0))), CanvasErrorCode.DUPLICATE_INPUT);
    }

    @Test
    void detectsCyclesInExecutionEdgesOnly() {
        CanvasNode first = node(CanvasNodeKind.TEXT_COMBINE, config());
        CanvasNode second = node(CanvasNodeKind.TEXT_COMBINE, config());
        rejects(graph(List.of(first, second), List.of(edge(first, "text", second, "texts", 0),
                edge(second, "text", first, "texts", 0))), CanvasErrorCode.EXECUTION_CYCLE);
        List<CanvasEdge> references = List.of(reference(first, second), reference(second, first));
        assertThat(validator.validate(graph(List.of(first, second), references)).topologicalNodeIds()).hasSize(2);
    }

    @Test
    void rejectsMissingEndpointsAndUnknownPorts() {
        CanvasNode text = node(CanvasNodeKind.TEXT, config());
        CanvasNode image = node(CanvasNodeKind.IMAGE_GENERATION, config());
        rejects(graph(List.of(text), List.of(edge(text, "text", image, "prompt", 0))), CanvasErrorCode.INVALID_GRAPH);
        rejects(graph(List.of(text, image), List.of(edge(text, "text", image, "unknown", 0))), CanvasErrorCode.INVALID_PORT);
    }

    @Test
    void rejectsBusinessReferenceOnUnrelatedNodeKinds() {
        CanvasNode image = node(CanvasNodeKind.IMAGE, config());
        CanvasNode invalid = new CanvasNode(image.id(), image.kind(), 1, image.title(), image.layout(), image.config(),
                new CanvasResourceReference(CanvasValueType.SHOT, 1L, 0L), 0, 0);
        rejects(graph(List.of(invalid), List.of()), CanvasErrorCode.INVALID_RESOURCE_REFERENCE);
    }

    @Test
    void requiresProjectAndMatchingBusinessResourceType() {
        CanvasNode shot = resourceNode(CanvasNodeKind.SHOT, CanvasValueType.SHOT);
        rejects(graph(List.of(shot), List.of()), CanvasErrorCode.INVALID_RESOURCE_REFERENCE);
        validator.validate(new CanvasGraph(1, 1L, List.of(shot), List.of()));
        CanvasNode wrong = resourceNode(CanvasNodeKind.SHOT, CanvasValueType.SCRIPT_SCENE);
        rejects(new CanvasGraph(1, 1L, List.of(wrong), List.of()), CanvasErrorCode.INVALID_RESOURCE_REFERENCE);
    }

    @Test
    void rejectsUnknownConfigurationAndInvalidDynamicPortTypes() {
        rejects(graph(List.of(node(CanvasNodeKind.IMAGE_GENERATION, config().put("apiKey", "test-value"))), List.of()),
                CanvasErrorCode.INVALID_CONFIG);
        rejects(graph(List.of(node(CanvasNodeKind.SELECTION, config().put("valueType", "unknown"))), List.of()),
                CanvasErrorCode.INVALID_CONFIG);
        rejects(graph(List.of(node(CanvasNodeKind.VIDEO_GENERATION, config().put("count", 0))), List.of()),
                CanvasErrorCode.INVALID_CONFIG);
        rejects(graph(List.of(node(CanvasNodeKind.IMAGE_GENERATION, config().put("modelId", "1"))), List.of()),
                CanvasErrorCode.INVALID_CONFIG);
    }

    @Test
    void rejectsGroupCyclesAndNonGroupParents() {
        String firstId = UUID.randomUUID().toString();
        String secondId = UUID.randomUUID().toString();
        CanvasNode first = groupedNode(firstId, CanvasNodeKind.GROUP, secondId);
        CanvasNode second = groupedNode(secondId, CanvasNodeKind.GROUP, firstId);
        rejects(graph(List.of(first, second), List.of()), CanvasErrorCode.INVALID_GROUP);
        rejects(graph(List.of(groupedNode(firstId, CanvasNodeKind.TEXT, secondId),
                groupedNode(secondId, CanvasNodeKind.TEXT, null)), List.of()), CanvasErrorCode.INVALID_GROUP);
    }

    @Test
    void rejectsFutureSchemasDuplicateIdentifiersAndNonFiniteCoordinates() {
        CanvasNode text = node(CanvasNodeKind.TEXT, config());
        rejects(new CanvasGraph(2, null, List.of(text), List.of()), CanvasErrorCode.UNSUPPORTED_SCHEMA);
        rejects(graph(List.of(text, text), List.of()), CanvasErrorCode.INVALID_GRAPH);
        CanvasNode invalid = new CanvasNode(text.id(), text.kind(), 1, text.title(),
                new CanvasNodeLayout(Double.NaN, 0, null, null, null, false, false), config(), null, 0, 0);
        rejects(graph(List.of(invalid), List.of()), CanvasErrorCode.INVALID_GRAPH);
    }

    private CanvasNode node(CanvasNodeKind kind, ObjectNode config) {
        return new CanvasNode(UUID.randomUUID().toString(), kind, 1, kind.code(),
                new CanvasNodeLayout(0, 0, null, null, null, false, false), config, null, 0, 0);
    }

    private CanvasNode resourceNode(CanvasNodeKind kind, CanvasValueType type) {
        CanvasNode base = node(kind, config());
        return new CanvasNode(base.id(), kind, 1, base.title(), base.layout(), base.config(),
                new CanvasResourceReference(type, 1L, 0L), 0, 0);
    }

    private CanvasNode groupedNode(String id, CanvasNodeKind kind, String parentId) {
        return new CanvasNode(id, kind, 1, kind.code(),
                new CanvasNodeLayout(0, 0, null, null, parentId, false, false), config(), null, 0, 0);
    }

    private CanvasGraph graph(List<CanvasNode> nodes, List<CanvasEdge> edges) {
        return new CanvasGraph(1, null, nodes, edges);
    }

    private CanvasEdge edge(CanvasNode source, String sourcePort, CanvasNode target, String targetPort, int index) {
        return new CanvasEdge(UUID.randomUUID().toString(), CanvasEdgeKind.EXECUTION,
                source.id(), sourcePort, target.id(), targetPort, index, null, 0);
    }

    private CanvasEdge reference(CanvasNode source, CanvasNode target) {
        return new CanvasEdge(UUID.randomUUID().toString(), CanvasEdgeKind.REFERENCE,
                source.id(), null, target.id(), null, null, null, 0);
    }

    private ObjectNode config() { return json.createObjectNode(); }

    private void rejects(CanvasGraph graph, CanvasErrorCode code) {
        assertThatThrownBy(() -> validator.validate(graph)).isInstanceOfSatisfying(CanvasException.class,
                failure -> assertThat(failure.getErrorCode()).isEqualTo(code));
    }
}
