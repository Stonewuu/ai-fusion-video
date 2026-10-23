package com.stonewu.fusion.service.canvas.model;

public record CanvasNodeLayout(double x, double y, Double width, Double height,
                               String parentId, boolean locked, boolean collapsed) {
}
