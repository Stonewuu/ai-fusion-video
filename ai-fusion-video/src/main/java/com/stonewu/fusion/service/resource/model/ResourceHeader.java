package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;

public record ResourceHeader(CanvasValueType resourceType, long resourceId, Long projectId, int version,
                             boolean deleted) {
}
