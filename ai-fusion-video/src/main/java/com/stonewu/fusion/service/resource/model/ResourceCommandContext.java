package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasActor;

import java.util.Objects;

/**
 * 身份和来源由 HTTP/Agent/后台任务边界构造，不能从模型参数或 HTTP 正文接受。
 */
public record ResourceCommandContext(CanvasActor actor, String origin, String canvasId,
                                     String agentRunId, String toolCallId, String ownerInstanceId, Long ownerEpoch) {
    public ResourceCommandContext {
        Objects.requireNonNull(actor);
        if (!"user".equals(origin) && !"agent".equals(origin) && !"system".equals(origin)) {
            throw new IllegalArgumentException("业务命令来源无效");
        }
        if ("agent".equals(
                origin) && (agentRunId == null || agentRunId.isBlank() || toolCallId == null || toolCallId.isBlank()
                || ownerInstanceId == null || ownerInstanceId.isBlank() || ownerEpoch == null || ownerEpoch <= 0)) {
            throw new IllegalArgumentException("智能体命令必须携带运行和工具调用标识");
        }
    }

    public static ResourceCommandContext user(CanvasActor actor, String canvasId) {
        return new ResourceCommandContext(actor, "user", canvasId, null, null, null, null);
    }
}
