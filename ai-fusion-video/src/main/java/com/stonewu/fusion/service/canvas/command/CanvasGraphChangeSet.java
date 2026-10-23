package com.stonewu.fusion.service.canvas.command;

import com.stonewu.fusion.service.canvas.model.CanvasEdge;
import com.stonewu.fusion.service.canvas.model.CanvasNode;

import java.util.List;

public record CanvasGraphChangeSet(List<CanvasNode> upsertedNodes, List<String> deletedNodeIds,
                                   List<CanvasEdge> upsertedEdges, List<String> deletedEdgeIds) {
    public CanvasGraphChangeSet {
        upsertedNodes = List.copyOf(upsertedNodes);
        deletedNodeIds = List.copyOf(deletedNodeIds);
        upsertedEdges = List.copyOf(upsertedEdges);
        deletedEdgeIds = List.copyOf(deletedEdgeIds);
    }
}
