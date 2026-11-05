package com.stonewu.fusion.service.ai.tool;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.text.LongToolResultStore;
import com.stonewu.fusion.service.ai.text.TextPages;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class ReadToolResultToolExecutor implements ToolExecutor {
    public static final String NAME = "read_tool_result";
    public static final String SCHEMA = """
            {
              "type": "object",
              "properties": {
                "resultId": {"type": "string", "description": "超长工具结果返回的结果编号，不是文件路径"},
                "offset": {"type": "integer", "minimum": 0, "description": "首次传 0，继续读取使用 nextOffset"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 64000, "description": "每段最多字符数；省略时按模型配置容量分配，实际返回量受该容量约束，最多 64000 字符"},
                "contentVersion": {"type": "string", "description": "继续读取时传入上一段返回的内容版本"}
              },
              "required": ["resultId"],
              "additionalProperties": false
            }
            """;
    private final LongToolResultStore results;
    private final CanvasJson json;

    public ReadToolResultToolExecutor(LongToolResultStore results, CanvasJson json) {
        this.results = results;
        this.json = json;
    }

    @Override
    public String getToolName() {
        return NAME;
    }

    @Override
    public String getDisplayName() {
        return "分段读取工具结果";
    }

    @Override
    public String getToolDescription() {
        return "读取当前会话中已完整保存的长工具结果。使用结果给出的 resultId，从 offset=0 开始，"
                + "按 nextOffset 连续读取至 hasMore=false 后才能判断全文；不支持文件路径或其他会话。";
    }

    @Override
    public String getParametersSchema() {
        return SCHEMA;
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public boolean isConcurrencySafe() {
        return true;
    }

    @Override
    public String execute(String input, ToolExecutionContext context) {
        try {
            ObjectNode params = json.read(input, ObjectNode.class);
            String resultId = params.path("resultId").asText();
            var page = results.read(context.getUserId().toString(), context.getAgentSessionId(), resultId,
                    TextPages.Request.from(params, context.getModel().getContextWindow()));
            ObjectNode response = json.object(page);
            response.put("status", "success").put("resultId", resultId);
            return json.write(response);
        } catch (CanvasException failure) {
            return json.write(Map.of("status", "error", "message", failure.getMessage(),
                    "errorCode", failure.getErrorCode()));
        }
    }
}
