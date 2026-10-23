package com.stonewu.fusion.service.canvas.run;

import com.stonewu.fusion.service.canvas.model.CanvasEdgeKind;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 只有实际进入分集合成的资源路径才冻结镜头视频；编辑分集说明不要求已有成片。
 */
public final class CanvasRunSourceUsage {
    private CanvasRunSourceUsage() {
    }

    public static Set<String> ports(CanvasGraph graph, String sourceId, Set<String> selected) {
        Set<String> used = new HashSet<>();
        Map<String, CanvasNode> nodes = graph.nodes().stream()
                .collect(Collectors.toMap(CanvasNode::id, Function.identity()));
        graph.edges().stream()
                .filter(edge -> edge.kind() == CanvasEdgeKind.EXECUTION && edge.sourceNodeId().equals(sourceId)
                        && selected.contains(edge.targetNodeId())).forEach(edge -> {
                    used.add(edge.sourcePort());
                    if (nodes.get(sourceId).kind() == CanvasNodeKind.STORYBOARD_EPISODE && edge.sourcePort().equals("episode")
                            && composition(graph, nodes, selected, edge.targetNodeId(), edge.targetPort(), new HashSet<>())) {
                        used.add("_canvasVideos");
                    }
                });
        return used;
    }

    private static boolean composition(CanvasGraph graph, Map<String, CanvasNode> nodes, Set<String> selected,
                                       String id, String input, Set<String> visited) {
        if (!visited.add(id + ":" + input)) {
            return false;
        }
        var node = nodes.get(id);
        if (node.kind() == CanvasNodeKind.VIDEO_COMPOSE) {
            return input.equals("episode");
        }
        if (node.kind() == CanvasNodeKind.FOREACH) {
            return input.equals("items")
                    && node.config().path("operationKind").asText().equals("videoCompose") && node.config()
                    .path("inputPort").asText().equals("episode");
        }
        String output = switch (node.kind()) {
            case COLLECTION -> "items";
            case SELECTION -> "selected";
            case WRITEBACK -> input.equals("target") ? "resource" : null;
            default -> null;
        };
        if (output == null) {
            return false;
        }
        return graph.edges().stream()
                .filter(edge -> edge.kind() == CanvasEdgeKind.EXECUTION && edge.sourceNodeId().equals(id)
                        && output.equals(edge.sourcePort()) && selected.contains(edge.targetNodeId()))
                .anyMatch(edge -> composition(graph, nodes, selected, edge.targetNodeId(), edge.targetPort(),
                        new HashSet<>(visited)));
    }
}
