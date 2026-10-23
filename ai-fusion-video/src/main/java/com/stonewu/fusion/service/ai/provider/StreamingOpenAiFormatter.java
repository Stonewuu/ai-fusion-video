package com.stonewu.fusion.service.ai.provider;

import io.agentscope.core.model.ChatResponse;
import io.agentscope.extensions.model.openai.dto.OpenAIResponse;
import io.agentscope.extensions.model.openai.formatter.OpenAIChatFormatter;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

/**
 * 单次请求内按协议 index 还原工具身份，避免并行工具的后续片段被归入最后一个工具。
 */
final class StreamingOpenAiFormatter extends OpenAIChatFormatter {
    private final Map<Integer, ToolIdentity> toolCalls = new HashMap<>();

    @Override
    public ChatResponse parseResponse(OpenAIResponse response, Instant started) {
        var choice = response.getFirstChoice();
        if (choice != null && choice.getDelta() != null && choice.getDelta().getToolCalls() != null) {
            for (var call : choice.getDelta().getToolCalls()) {
                if (call.getIndex() == null || call.getFunction() == null) {
                    throw new IllegalStateException("模型流式工具调用缺少 index 或 function");
                }
                var identity = toolCalls.computeIfAbsent(call.getIndex(), ignored -> new ToolIdentity());
                if (call.getId() != null && !call.getId().isBlank()) {
                    identity.id = call.getId();
                }
                if (call.getFunction().getName() != null && !call.getFunction().getName().isBlank()) {
                    identity.name = call.getFunction().getName();
                }
                if (identity.id == null || identity.name == null) {
                    throw new IllegalStateException("模型工具参数片段先于调用身份到达");
                }
                call.setId(identity.id);
                call.getFunction().setName(identity.name);
            }
        }
        return super.parseResponse(response, started);
    }

    private static final class ToolIdentity {
        private String id;
        private String name;
    }
}
