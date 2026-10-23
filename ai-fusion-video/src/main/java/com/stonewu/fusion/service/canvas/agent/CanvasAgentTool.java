package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.ToolPermissionRisk;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasException;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.function.BiFunction;

/**
 * 画布工具使用统一错误回执、风险分类和大小边界，不接受身份或原始执行脚本。
 */
public class CanvasAgentTool implements ToolExecutor {
    private final String name;
    private final String label;
    private final String description;
    private final ObjectNode schema;
    private final boolean readOnly;
    private final CanvasJson json;
    private final BiFunction<ObjectNode, ToolExecutionContext, Object> handler;

    public CanvasAgentTool(String name, String label, String description, ObjectNode schema, boolean readOnly,
                           CanvasJson json, BiFunction<ObjectNode, ToolExecutionContext, Object> handler) {
        this.name = name;
        this.label = label;
        this.description = description;
        this.schema = schema.deepCopy();
        this.readOnly = readOnly;
        this.json = json;
        this.handler = handler;
    }

    @Override
    public String getToolName() {
        return name;
    }

    @Override
    public String getDisplayName() {
        return label;
    }

    @Override
    public String getToolDescription() {
        return description;
    }

    @Override
    public String getParametersSchema() {
        return json.write(schema);
    }

    @Override
    public boolean isReadOnly() {
        return readOnly;
    }

    @Override
    public boolean isConcurrencySafe() {
        return readOnly;
    }

    @Override
    public boolean mayRequireHighRiskApproval() {
        return name.equals("apply_canvas_operations") || name.equals("edit_canvas_resource") || name.equals(
                "bind_canvas_output") || name.equals("confirm_canvas_writeback")
                || name.equals("run_canvas_nodes") || name.equals("undo_canvas_command") || name.equals(
                "review_canvas_batch");
    }

    @Override
    public ToolPermissionRisk getPermissionRisk(Map<String, Object> input) {
        if (readOnly) {
            return ToolPermissionRisk.READ_ONLY;
        }
        if (name.equals("run_canvas_nodes") || name.equals("undo_canvas_command")) {
            return ToolPermissionRisk.HIGH_RISK;
        }
        JsonNode value = json.object(input);
        if (name.equals("review_canvas_batch") && "RETRY_FAILED".equals(value.path("action").asText())) {
            return ToolPermissionRisk.HIGH_RISK;
        }
        if (name.equals("apply_canvas_operations")) {
            for (JsonNode operation : value.path("operations")) {
                if (operation.path("type").asText().endsWith(".delete")) {
                    return ToolPermissionRisk.HIGH_RISK;
                }
            }
        }
        if (name.equals("edit_canvas_resource") && (Set.of("delete", "replaceSource")
                .contains(value.path("action").asText())
                || value.path("fields").has("rawContent"))) {
            return ToolPermissionRisk.HIGH_RISK;
        }
        if (name.equals("bind_canvas_output") && "script:rawContent".equals(value.path("binding").asText())) {
            return ToolPermissionRisk.HIGH_RISK;
        }
        if (name.equals("confirm_canvas_writeback") && "APPROVE".equals(value.path("action").asText())) {
            return ToolPermissionRisk.HIGH_RISK;
        }
        return ToolPermissionRisk.EDIT;
    }

    @Override
    public String execute(String input, ToolExecutionContext context) {
        try {
            if (input == null || input.length() > 512 * 1024) {
                throw new BusinessException(400, "工具参数超过 512 KiB，请拆分修改");
            }
            ObjectNode arguments;
            try {
                arguments = json.read(input, ObjectNode.class);
            } catch (IllegalStateException invalid) {
                throw new BusinessException(400, "工具参数必须为 JSON 对象");
            }
            if (arguments == null) {
                throw new BusinessException(400, "工具参数不能为空");
            }
            CanvasAgentSchema.validate(arguments, schema, "参数");
            return json.write(handler.apply(arguments, context));
        } catch (BusinessException failure) {
            var result = new LinkedHashMap<String, Object>();
            result.put("status", "error");
            result.put("code", failure.getCode());
            result.put("message", failure.getMessage());
            if (failure instanceof CanvasException canvas) {
                result.put("errorCode", canvas.getErrorCode());
                result.put("conflicts", canvas.getDetails());
            }
            return json.write(result);
        }
    }
}
