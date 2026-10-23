package com.stonewu.fusion.service.canvas.model;

import java.util.List;

public record CanvasGraphValidation(List<String> topologicalNodeIds) {
    public CanvasGraphValidation {
        topologicalNodeIds = List.copyOf(topologicalNodeIds);
    }
}
