package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.TextBlock;

import java.util.List;

/**
 * null 明确表示继续等待；具体工具仅允许读取已授权的内部任务结果。
 */
public interface CanvasExternalTool extends ToolExecutor {
    Object poll(ObjectNode input, CanvasAgentScope scope, CanvasActor actor, AiModel model);

    default List<ContentBlock> projectOutput(String result) {
        return List.of(TextBlock.builder().text(result).build());
    }
}
