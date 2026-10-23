package com.stonewu.fusion.service.ai.tool.project;

import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceToolRequests;
import com.stonewu.fusion.service.resource.model.ResourceDeleteCommand;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * 删除项目工具。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DeleteProjectToolExecutor implements ToolExecutor {

    private final ResourceCommandService commands;
    private final CanvasJson json;

    @Override
    public String getToolName() {
        return "delete_project";
    }

    @Override
    public String getDisplayName() {
        return "删除项目";
    }

    @Override
    public String getToolDescription() {
        return "按查询得到的 expectedVersion 删除当前用户管理的项目及其剧本、分镜和资产，属于高风险操作。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "projectId": {"type": "integer", "description": "要删除的项目ID"},
                        "expectedVersion": {"type": "integer", "minimum": 0, "description": "查询得到的项目版本"}
                    },
                    "required": ["projectId", "expectedVersion"],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            ObjectNode params = json.read(toolInput, ObjectNode.class);
            var id = params.get("projectId");
            var version = params.get("expectedVersion");
            if (id == null || !id.isIntegralNumber() || !id.canConvertToLong() || id.longValue() <= 0) {
                return error("projectId 无效");
            }
            if (version == null || !version.isIntegralNumber() || !version.canConvertToInt() || version.intValue() < 0) {
                return error("expectedVersion 必填");
            }
            var actor = context.requireMutationContext();
            var receipt = commands.execute(new ResourceDeleteCommand(ResourceToolRequests.requestId(json, actor),
                    CanvasValueType.PROJECT, id.longValue(), version.intValue()), actor);
            return json.write(Map.of("status", "success", "deletedProjectId", id.longValue(),
                    "commandId", receipt.commandId(), "changes", receipt.changes()));
        } catch (Exception e) {
            log.error("删除项目失败", e);
            return error("删除项目失败: " + e.getMessage());
        }
    }

    private String error(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
