package com.stonewu.fusion.service.canvas.command;

public record CanvasCommandReceipt(String commandId, String requestId, String canvasId,
                                   long revision, long sequence, CanvasGraphChangeSet changes) {
}
