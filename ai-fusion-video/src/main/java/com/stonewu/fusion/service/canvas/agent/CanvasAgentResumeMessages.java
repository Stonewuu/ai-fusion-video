package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.ToolExecutorRegistry;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeTypes;
import io.agentscope.core.message.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 每条工具返回单独成消息，避免模型协议转换时只保留第一个工具结果。
 */
final class CanvasAgentResumeMessages {
    private CanvasAgentResumeMessages() {
    }

    static List<Msg> external(ObjectNode saved, ToolExecutorRegistry tools) {
        List<Msg> messages = new ArrayList<>();
        Set<String> ids = new HashSet<>();
        for (JsonNode entry : saved.path("results")) {
            String id = entry.path("toolCallId").asText();
            String name = entry.path("toolName").asText();
            if (id.isBlank() || !ids.add(id) || !entry.path("result").isObject()) {
                throw new IllegalStateException("恢复工具结果身份或内容无效");
            }
            var executor = tools.findExecutor(name);
            if (!(executor instanceof CanvasExternalTool) && !CanvasAgentNodeTypes.CHILD_TOOLS.contains(name)) {
                throw new IllegalStateException("恢复工具未登记");
            }
            var output = ToolResultBlock.builder().id(id).name(name)
                    .output(executor instanceof CanvasExternalTool external ? external.projectOutput(
                            entry.path("result").toString())
                            : List.of(TextBlock.builder().text(entry.path("result").toString()).build()))
                    .state("error".equals(entry.path("result").path("status")
                            .asText()) ? ToolResultState.ERROR : ToolResultState.SUCCESS).build();
            messages.add(Msg.builder().role(MsgRole.TOOL).content(List.of(output)).build());
        }
        if (messages.isEmpty()) {
            throw new IllegalStateException("恢复结果为空");
        }
        return List.copyOf(messages);
    }
}
