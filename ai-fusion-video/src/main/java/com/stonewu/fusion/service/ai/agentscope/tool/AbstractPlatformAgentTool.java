package com.stonewu.fusion.service.ai.agentscope.tool;

import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolResultState;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.tool.ToolBase;
import io.agentscope.core.tool.ToolCallParam;
import reactor.core.publisher.Mono;

/** Forces every platform AgentScope tool to implement the V2 asynchronous contract. */
public abstract class AbstractPlatformAgentTool extends ToolBase {

    protected AbstractPlatformAgentTool(Builder builder) {
        super(builder);
    }

    @Override
    public abstract Mono<ToolResultBlock> callAsync(ToolCallParam param);

    protected final ToolResultBlock textResult(ToolCallParam param, String result) {
        if (result == null) {
            throw new IllegalStateException("AgentScope tool returned null output: " + getName());
        }
        ToolUseBlock toolUse = requireToolUse(param);
        return ToolResultBlock.text(result).withIdAndName(toolUse.getId(), toolUse.getName());
    }

    protected final ToolResultBlock errorResult(ToolCallParam param, String error) {
        if (error == null || error.isBlank()) {
            throw new IllegalArgumentException(
                    "AgentScope tool error must not be blank: " + getName());
        }
        ToolUseBlock toolUse = requireToolUse(param);
        return ToolResultBlock.error(error)
                .withState(ToolResultState.ERROR)
                .withIdAndName(toolUse.getId(), toolUse.getName());
    }

    protected final ToolUseBlock requireToolUse(ToolCallParam param) {
        if (param == null || param.getToolUseBlock() == null) {
            throw new IllegalArgumentException("AgentScope tool call identity is required: " + getName());
        }
        ToolUseBlock toolUse = param.getToolUseBlock();
        if (toolUse.getId() == null || toolUse.getId().isBlank()
                || toolUse.getName() == null || toolUse.getName().isBlank()) {
            throw new IllegalArgumentException("AgentScope tool call identity is incomplete: " + getName());
        }
        return toolUse;
    }
}
