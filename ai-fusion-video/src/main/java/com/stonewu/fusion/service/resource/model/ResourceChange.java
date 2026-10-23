package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;

import java.util.List;

public record ResourceChange(String eventId, CanvasValueType resourceType, long resourceId, Long projectId,
                             int version, boolean deleted, List<String> changedFields) {
    public ResourceChange {
        changedFields = List.copyOf(changedFields);
    }
}
