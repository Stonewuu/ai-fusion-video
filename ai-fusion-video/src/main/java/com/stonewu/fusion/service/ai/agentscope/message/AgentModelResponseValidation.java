package com.stonewu.fusion.service.ai.agentscope.message;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.model.ChatResponse;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 正文直接转发，只累计工具参数；流完成前校验，阻止 SDK 把残缺参数当成空对象执行。
 */
final class AgentModelResponseValidation {
    private static final ObjectMapper JSON = new ObjectMapper()
            .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);

    private final Map<String, Arguments> tools = new LinkedHashMap<>();
    private String lastTool;
    private String finishReason;

    void accept(ChatResponse response) {
        String reason = response.getFinishReason();
        if (reason != null && !reason.isBlank() && !"null".equalsIgnoreCase(reason)) {
            finishReason = reason;
        }
        for (var block : response.getContent()) {
            if (!(block instanceof ToolUseBlock tool)) {
                continue;
            }
            String key = tool.getId();
            if (key == null || key.isBlank()) {
                key = lastTool;
            }
            if (key == null) {
                throw new IllegalStateException("模型工具参数片段缺少所属调用编号");
            }
            lastTool = key;
            Arguments arguments = tools.computeIfAbsent(key, ignored -> new Arguments());
            if (tool.getName() != null && !tool.getName().startsWith("__") && !tool.getName().isBlank()) {
                arguments.name = tool.getName();
            }
            if (tool.getContent() != null) {
                arguments.raw.append(tool.getContent());
            }
        }
    }

    void finish() {
        if (finishReason == null) {
            throw new IllegalStateException("模型连接在返回完成标记前中断，未执行本轮工具");
        }
        if ("length".equalsIgnoreCase(finishReason) || "max_tokens".equalsIgnoreCase(finishReason)) {
            throw new BusinessException(502, "模型输出达到长度上限，未执行本轮工具。请提高模型输出上限或拆分要求后重试。");
        }
        for (Arguments arguments : tools.values()) {
            if (arguments.raw.isEmpty()) {
                continue;
            }
            try {
                var value = JSON.readTree(arguments.raw.toString());
                if (value == null || !value.isObject()) {
                    throw invalid(arguments.name);
                }
            } catch (JsonProcessingException malformed) {
                throw invalid(arguments.name);
            }
        }
    }

    private static BusinessException invalid(String toolName) {
        return new BusinessException(502, "模型返回的工具参数不完整或格式错误，未执行：" + toolName);
    }

    private static final class Arguments {
        private String name;
        private final StringBuilder raw = new StringBuilder();
    }
}
