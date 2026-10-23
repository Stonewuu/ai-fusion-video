package com.stonewu.fusion.service.canvas.model;

public record CanvasSnapshot(CanvasDocument document, CanvasGraph graph, long lastSequence,
                             CanvasPermissions permissions) {
}
