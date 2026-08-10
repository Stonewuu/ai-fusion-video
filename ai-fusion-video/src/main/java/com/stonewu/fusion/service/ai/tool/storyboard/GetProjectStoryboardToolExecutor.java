package com.stonewu.fusion.service.ai.tool.storyboard;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/** 查询项目唯一分镜容器工具。 */
@Component
@RequiredArgsConstructor
@Slf4j
public class GetProjectStoryboardToolExecutor implements ToolExecutor {

    private final StoryboardService storyboardService;
    private final ProjectService projectService;

    @Override
    public String getToolName() {
        return "get_project_storyboard";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询项目分镜";
    }

    @Override
    public String getToolDescription() {
        return """
                查询指定项目唯一的分镜容器。
                返回分镜ID、项目ID、关联剧本ID、描述和总时长等基本信息。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "projectId": {
                            "type": "integer",
                            "description": "项目ID（必填）"
                        }
                    },
                    "required": ["projectId"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long projectId = params.getLong("projectId");
            if (projectId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 projectId").toString();
            }

            Long userId = context.getUserId();

            if (!projectService.canAccessProject(projectId, userId)) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "无权访问该项目").toString();
            }

            Storyboard storyboard = storyboardService.getByProjectId(projectId);
            if (storyboard == null) {
                return JSONUtil.createObj()
                        .set("projectId", projectId)
                        .set("status", "empty")
                        .set("message", "项目分镜容器不存在")
                        .toString();
            }

            return JSONUtil.createObj()
                    .set("projectId", projectId)
                    .set("storyboardId", storyboard.getId())
                    .set("scriptId", storyboard.getScriptId())
                    .set("description", storyboard.getDescription())
                    .set("totalDuration", storyboard.getTotalDuration())
                    .toString();
        } catch (Exception e) {
            log.error("查询项目分镜失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }
}
