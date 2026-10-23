package com.stonewu.fusion.service.ai.tool.project;

import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceToolRequests;
import com.stonewu.fusion.service.resource.model.ResourceCommand;
import com.stonewu.fusion.service.resource.model.ResourceCreateCommand;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * 创建或更新项目工具。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SaveProjectToolExecutor implements ToolExecutor {

    private final ResourceCommandService resourceCommands;
    private final CanvasJson json;

    @Override
    public String getToolName() {
        return "save_project";
    }

    @Override
    public String getDisplayName() {
        return "保存项目";
    }

    @Override
    public String getToolDescription() {
        return "创建项目，或传入 projectId 和查询得到的 expectedVersion 更新项目。未传 projectId 时 name 必填。版本冲突后须重新读取并决定修改。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "projectId": {"type": "integer", "description": "项目ID；不传表示创建"},
                        "expectedVersion": {"type": "integer", "minimum": 0, "description": "更新必填，来自项目查询结果的 version；不得猜测"},
                        "name": {"type": "string", "description": "项目名称；创建时必填"},
                        "description": {"type": "string", "description": "项目描述"},
                        "coverUrl": {"type": "string", "description": "封面URL"},
                        "properties": {"type": "string", "description": "项目扩展属性JSON"},
                        "artStyle": {"type": "string", "description": "画风名称"},
                        "artStyleDescription": {"type": "string", "description": "画风描述"},
                        "artStyleImagePrompt": {"type": "string", "description": "画风图片提示词"},
                        "artStyleImageUrl": {"type": "string", "description": "画风参考图URL"},
                        "status": {"type": "integer", "description": "项目状态"}
                    },
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            var mutation = context.requireMutationContext();
            ObjectNode fields = json.read(toolInput, ObjectNode.class);
            var project = fields.remove("projectId");
            var version = fields.remove("expectedVersion");
            if (fields.has("properties") && fields.get("properties").isTextual()) {
                fields.set("properties", json.read(fields.get("properties").textValue(), ObjectNode.class));
            }
            String requestId = ResourceToolRequests.requestId(json, mutation);
            ResourceCommand command;
            String operation;
            if (project == null || project.isNull()) {
                if (version != null) {
                    return error("创建项目不接受 expectedVersion");
                }
                command = new ResourceCreateCommand(requestId, CanvasValueType.PROJECT, null, fields);
                operation = "created";
            } else {
                if (!project.isIntegralNumber() || !project.canConvertToLong() || project.longValue() <= 0) {
                    return error("projectId 无效");
                }
                if (version == null || !version.isIntegralNumber() || !version.canConvertToInt() || version.intValue() < 0) {
                    return error("更新项目必须提供查询得到的 expectedVersion");
                }
                command = new ResourcePatchCommand(requestId, CanvasValueType.PROJECT, project.longValue(),
                        version.intValue(), fields);
                operation = "updated";
            }
            var receipt = resourceCommands.execute(command, mutation);
            return json.write(Map.of("status", "success", "operation", operation, "project", receipt.result(),
                    "commandId", receipt.commandId(), "changes", receipt.changes()));
        } catch (Exception failure) {
            log.error("保存项目失败", failure);
            return error("保存项目失败: " + failure.getMessage());
        }
    }

    private String error(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
