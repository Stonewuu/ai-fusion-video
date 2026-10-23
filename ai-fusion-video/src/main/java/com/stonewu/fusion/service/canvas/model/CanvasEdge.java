package com.stonewu.fusion.service.canvas.model;

public record CanvasEdge(String id, CanvasEdgeKind kind, String sourceNodeId, String sourcePort,
                         String targetNodeId, String targetPort, Integer inputIndex,
                         CanvasOutputReference output, long version) {
}
