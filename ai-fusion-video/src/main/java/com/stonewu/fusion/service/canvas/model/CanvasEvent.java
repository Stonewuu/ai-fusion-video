package com.stonewu.fusion.service.canvas.model;

import com.fasterxml.jackson.databind.node.ObjectNode;

import java.time.LocalDateTime;

public record CanvasEvent(String id, String canvasId, long sequence, String commandId,
                          String type, ObjectNode payload, LocalDateTime createdAt) {
}
