package com.stonewu.fusion.service.canvas.agent;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasSnapshot;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.UUID;

@Component
@RequiredArgsConstructor
public class CanvasAgentAccess {
    private final CanvasQueryService canvases;

    public Session require(ToolExecutionContext tool) {
        var scope = tool.getCanvasScope();
        var context = tool.requireMutationContext();
        if (scope == null) {
            throw new BusinessException(409, "请在画布中发送消息以绑定操作范围");
        }
        if (!Objects.equals(scope.canvasId(), context.canvasId())) {
            throw new BusinessException(403, "画布工具作用域不一致");
        }
        var snapshot = canvases.snapshot(scope.canvasId(), context.actor());
        if (!Objects.equals(scope.projectId(), snapshot.document().projectId())) {
            throw new BusinessException(403, "画布项目范围已变化");
        }
        return new Session(scope, context, snapshot);
    }

    public record Session(CanvasAgentScope scope, ResourceCommandContext context, CanvasSnapshot snapshot) {
        public CanvasNode node(String id) {
            return snapshot.graph().nodes().stream().filter(node -> node.id().equals(id)).findFirst()
                    .orElseThrow(() -> new BusinessException(404, "节点不在当前画布中"));
        }

        public String requestId() {
            return "agent:" + context.agentRunId() + ":" + id("invocation");
        }

        public String id(String clientRef) {
            return UUID.nameUUIDFromBytes(
                    (context.agentRunId() + "\u0000" + context.toolCallId() + "\u0000" + clientRef)
                            .getBytes(StandardCharsets.UTF_8)).toString();
        }
    }
}
