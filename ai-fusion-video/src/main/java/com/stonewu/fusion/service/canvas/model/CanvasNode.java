package com.stonewu.fusion.service.canvas.model;

import com.fasterxml.jackson.databind.node.ObjectNode;

public record CanvasNode(String id, CanvasNodeKind kind, int schemaVersion, String title,
                         CanvasNodeLayout layout, ObjectNode config,
                         CanvasResourceReference resource, long configVersion, long layoutVersion) {
}
