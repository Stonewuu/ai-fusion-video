package com.stonewu.fusion.service.canvas.model;

import java.util.List;

public record CanvasGraph(int schemaVersion, Long projectId,
                          List<CanvasNode> nodes, List<CanvasEdge> edges) {
}
