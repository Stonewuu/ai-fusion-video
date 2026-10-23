package com.stonewu.fusion.service.canvas.run.media;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Ports;

public record CanvasMediaInput(int schemaVersion, String nodeRunId, CanvasNodeKind operation,
                               ObjectNode config, Ports inputs, ObjectNode environment) {
}
