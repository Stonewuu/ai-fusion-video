package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation;
import com.stonewu.fusion.service.canvas.model.CanvasEdge;
import com.stonewu.fusion.service.canvas.model.CanvasEdgeKind;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasNodeLayout;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;

class CanvasAgentLayoutTests {
    private final CanvasAgentLayout layout = new CanvasAgentLayout(null, null);

    @Test void resultsAndNotesFollowTheSourceChainWithoutChangingGraphEdges() {
        var graph = new CanvasGraph(1, null, List.of(node("result"), node("a"), node("b"), node("note")), List.of(
                edge("a", "b", CanvasEdgeKind.EXECUTION), edge("b", "result", CanvasEdgeKind.REFERENCE), edge("result", "note", CanvasEdgeKind.ANNOTATION)));
        var positions = arrange(graph);
        for (var pair : List.of(List.of("a", "b"), List.of("b", "result"), List.of("result", "note"))) {
            assertThat(positions.get(pair.get(0)).x()).isLessThan(positions.get(pair.get(1)).x());
            assertThat(positions.get(pair.get(0)).y()).isEqualTo(positions.get(pair.get(1)).y());
        }
        assertThat(graph.edges()).hasSize(3);
    }

    @Test void annotationsThatPointBackDoNotOverrideTheExecutionOrder() {
        var graph = new CanvasGraph(1, null, List.of(node("a"), node("b"), node("c")), List.of(
                edge("c", "a", CanvasEdgeKind.ANNOTATION), edge("a", "b", CanvasEdgeKind.EXECUTION), edge("b", "c", CanvasEdgeKind.EXECUTION)));
        var positions = arrange(graph);
        assertThat(positions.get("a").x()).isLessThan(positions.get("b").x());
        assertThat(positions.get("b").x()).isLessThan(positions.get("c").x());
        assertThat(graph.edges()).hasSize(3);
    }

    private Map<String, CanvasNodeLayout> arrange(CanvasGraph graph) {
        return layout.layout(graph, graph.nodes().stream().map(CanvasNode::id).toList(), "arrange").stream()
                .map(operation -> (CanvasGraphOperation.LayoutNode) operation)
                .collect(Collectors.toMap(CanvasGraphOperation.LayoutNode::nodeId, CanvasGraphOperation.LayoutNode::layout));
    }
    private CanvasNode node(String id) {
        return new CanvasNode(id, CanvasNodeKind.TEXT, 1, id, new CanvasNodeLayout(0, 0, 300.0, 200.0, null, false, false), JsonNodeFactory.instance.objectNode(), null, 0, 0);
    }
    private CanvasEdge edge(String source, String target, CanvasEdgeKind kind) {
        return new CanvasEdge(source + target, kind, source, "text", target, "input", 0, null, 0);
    }
}
