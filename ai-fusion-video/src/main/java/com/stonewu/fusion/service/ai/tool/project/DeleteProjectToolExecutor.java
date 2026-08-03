package com.stonewu.fusion.service.ai.tool.project;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.project.ProjectService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/** 删除项目工具。 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DeleteProjectToolExecutor implements ToolExecutor {

    private final ProjectService projectService;

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
        return "删除当前用户可访问的项目。该操作会级联删除项目资产，属于高风险操作。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "projectId": {"type": "integer", "description": "要删除的项目ID"}
                    },
                    "required": ["projectId"],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long projectId = params.getLong("projectId");
            if (projectId == null) {
                return error("缺少 projectId");
            }
            Project project = projectService.getById(projectId);
            if (!projectService.canAccessProject(project, context.getUserId())) {
                return error("无权删除该项目");
            }
            projectService.delete(projectId);
            return JSONUtil.createObj()
                    .set("status", "success")
                    .set("deletedProjectId", projectId)
                    .toString();
        } catch (Exception e) {
            log.error("删除项目失败", e);
            return error("删除项目失败: " + e.getMessage());
        }
    }

    private String error(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
