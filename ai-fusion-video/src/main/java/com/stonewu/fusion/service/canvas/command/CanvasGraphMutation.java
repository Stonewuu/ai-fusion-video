package com.stonewu.fusion.service.canvas.command;

import com.stonewu.fusion.service.canvas.model.CanvasGraph;

public record CanvasGraphMutation(CanvasGraph graph, CanvasGraphChangeSet changes,
                                  CanvasGraphChangeSet previousValues) {
}
