package com.stonewu.fusion.service.canvas.agent;

import com.stonewu.fusion.service.canvas.model.CanvasResourceReference;

import java.util.List;

/**
 * 客户端只提供定位信息；正文、版本和权限必须由服务端重新读取。
 */
public record CanvasAgentContext(String canvasId, List<String> selectedNodeIds,
                                 List<String> visibleNodeIds, CanvasResourceReference focusedResource,
                                 Long lastSequence) {
}
