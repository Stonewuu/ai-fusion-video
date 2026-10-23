package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;

import java.util.Map;

public final class ResourceToolRequests {
    private ResourceToolRequests() {
    }

    public static String requestId(CanvasJson json, ResourceCommandContext context) {
        if (!"agent".equals(context.origin())) {
            throw new IllegalArgumentException("工具调用缺少可信智能体身份");
        }
        return "agent:" + json.hash(Map.of("run", context.agentRunId(), "call", context.toolCallId()));
    }
}
