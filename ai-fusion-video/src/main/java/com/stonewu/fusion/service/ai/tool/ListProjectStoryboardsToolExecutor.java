package com.stonewu.fusion.service.ai.tool;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * 列出项目分镜工具（list_project_storyboards）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ListProjectStoryboardsToolExecutor implements ToolExecutor {

    private final StoryboardService storyboardService;
    private final ProjectService projectService;

    @Override
    public String getToolName() {
        return "list_project_storyboards";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "列出项目分镜";
    }

    @Override
    public String getToolDescription() {
        return """
                列出指定项目下的所有分镜脚本。
                返回分镜ID、标题、描述、总时长等基本信息。
                如需查看分镜的详细条目，请使用 query_storyboard 工具。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "projectId": {
                            "type": "number",
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

            List<Storyboard> storyboards = storyboardService.listByProject(projectId);
            JSONArray list = new JSONArray();
            for (Storyboard sb : storyboards) {
                list.add(JSONUtil.createObj()
                        .set("storyboardId", sb.getId())
                        .set("title", sb.getTitle())
                        .set("description", sb.getDescription())
                        .set("totalDuration", sb.getTotalDuration()));
            }

            return JSONUtil.createObj()
                    .set("projectId", projectId)
                    .set("count", storyboards.size())
                    .set("storyboards", list).toString();
        } catch (Exception e) {
            log.error("列出项目分镜失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }
}
