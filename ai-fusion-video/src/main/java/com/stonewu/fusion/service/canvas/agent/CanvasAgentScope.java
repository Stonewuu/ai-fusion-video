package com.stonewu.fusion.service.canvas.agent;

import com.stonewu.fusion.service.canvas.model.CanvasResourceReference;

import java.util.List;

/**
 * 每轮运行固定的可信画布上下文；子任务和恢复只继承此记录。
 */
public record CanvasAgentScope(String canvasId, Long projectId, List<String> selectedNodeIds,
                               List<String> visibleNodeIds, CanvasResourceReference focusedResource,
                               long clientSequence, long admittedSequence, long admittedRevision,
                               String nodeRunId) {
    public CanvasAgentScope(String canvasId, Long projectId, List<String> selectedNodeIds,
                            List<String> visibleNodeIds, CanvasResourceReference focusedResource,
                            long clientSequence, long admittedSequence, long admittedRevision) {
        this(canvasId, projectId, selectedNodeIds, visibleNodeIds, focusedResource, clientSequence, admittedSequence,
                admittedRevision, null);
    }

    public CanvasAgentScope {
        if (canvasId == null || canvasId.isBlank()) {
            throw new IllegalArgumentException("画布上下文缺少标识");
        }
        selectedNodeIds = List.copyOf(selectedNodeIds);
        visibleNodeIds = List.copyOf(visibleNodeIds);
    }
}
