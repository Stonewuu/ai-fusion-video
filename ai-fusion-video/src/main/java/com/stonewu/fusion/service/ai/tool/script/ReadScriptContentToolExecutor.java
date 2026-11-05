package com.stonewu.fusion.service.ai.tool.script;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.text.TextPages;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.ScriptFields;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Set;

@Component
@RequiredArgsConstructor
public class ReadScriptContentToolExecutor implements ToolExecutor {
    public static final String NAME = "read_script_content";
    public static final String SCHEMA = """
            {
              "type": "object",
              "properties": {
                "resourceType": {"type": "string", "enum": ["script", "episode"], "description": "剧本或分集"},
                "resourceId": {"type": "integer", "minimum": 1, "description": "查询结果中的业务对象 ID"},
                "field": {"type": "string", "enum": ["rawContent", "content"], "description": "读取入口指定的正文字段"},
                "offset": {"type": "integer", "minimum": 0, "description": "首次传 0，继续读取使用 nextOffset"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 64000, "description": "每段最多字符数；省略时按模型配置容量分配，实际返回量受该容量约束，最多 64000 字符"},
                "contentVersion": {"type": "string", "description": "读取入口或上一段返回的内容版本，继续读取时必传"}
              },
              "required": ["resourceType", "resourceId", "field"],
              "additionalProperties": false
            }
            """;
    private final ResourceQueryService queries;
    private final CanvasJson json;

    @Override
    public String getToolName() {
        return NAME;
    }

    @Override
    public String getDisplayName() {
        return "分段读取剧本正文";
    }

    @Override
    public String getToolDescription() {
        return "按查询结果的 contentReads 读取剧本或分集正文。每段返回 contentVersion、nextOffset 和 hasMore。"
                + "完整解析时按 nextOffset 连续读取至 hasMore=false，不能把一段当作全文。";
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
            CanvasValueType type = switch (params.path("resourceType").asText()) {
                case "script" -> CanvasValueType.SCRIPT;
                case "episode" -> CanvasValueType.SCRIPT_EPISODE;
                default -> throw ScriptFields.invalid("resourceType 必须为 script 或 episode");
            };
            long id = ScriptFields.positiveId(params.get("resourceId"), "resourceId");
            String field = params.path("field").asText();
            if (!Set.of("rawContent", "content").contains(field)
                    || type == CanvasValueType.SCRIPT_EPISODE && field.equals("content")) {
                throw ScriptFields.invalid("此对象不支持指定的正文字段");
            }
            var request = TextPages.Request.from(params, context.getModel().getContextWindow());
            var snapshot = queries.get(type, id, context.requireMutationContext().actor());
            var value = snapshot.data().get(field);
            if (value == null || value.isNull()) {
                throw ScriptFields.invalid("此对象尚未保存指定正文");
            }
            if (!value.isTextual()) {
                throw ScriptFields.invalid("已保存的正文格式无效");
            }
            ObjectNode result = json.object(TextPages.read(value.textValue(), request));
            result.put("status", "success").put("resourceType", params.path("resourceType").asText())
                    .put("resourceId", id).put("field", field)
                    .set("resourceVersion", snapshot.data().get("version"));
            return json.write(result);
        } catch (CanvasException failure) {
            return json.write(Map.of("status", "error", "message", failure.getMessage(),
                    "errorCode", failure.getErrorCode()));
        }
    }
}
