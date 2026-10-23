package com.stonewu.fusion.service.canvas.collaboration;

import com.stonewu.fusion.service.canvas.model.CanvasEdge;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeLayout;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public final class CanvasGraphCopies {
    private CanvasGraphCopies() {
    }

    public static CanvasGraph copy(CanvasGraph source, String destination) {
        Map<String, String> ids = new HashMap<>();
        source.nodes().forEach(node -> ids.put(node.id(), id(destination, "node", node.id())));
        var nodes = source.nodes().stream().map(node -> {
            var layout = node.layout();
            return new CanvasNode(ids.get(node.id()), node.kind(), node.schemaVersion(), node.title(),
                    new CanvasNodeLayout(layout.x(), layout.y(), layout.width(), layout.height(),
                            layout.parentId() == null ? null : ids.get(layout.parentId()), layout.locked(),
                            layout.collapsed()),
                    node.config().deepCopy(), node.resource(), 0, 0);
        }).toList();
        var edges = source.edges().stream().map(edge -> new CanvasEdge(id(destination, "edge", edge.id()), edge.kind(),
                ids.get(edge.sourceNodeId()), edge.sourcePort(), ids.get(edge.targetNodeId()), edge.targetPort(),
                edge.inputIndex(), edge.output(), 0)).toList();
        return new CanvasGraph(source.schemaVersion(), source.projectId(), nodes, edges);
    }

    private static String id(String canvas, String kind, String source) {
        return UUID.nameUUIDFromBytes((canvas + ":" + kind + ":" + source).getBytes(StandardCharsets.UTF_8)).toString();
    }
}
