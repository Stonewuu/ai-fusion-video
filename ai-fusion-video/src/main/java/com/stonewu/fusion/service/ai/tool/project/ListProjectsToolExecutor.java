package com.stonewu.fusion.service.ai.tool.project;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * 列出用户项目工具（list_my_projects）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ListProjectsToolExecutor implements ToolExecutor {

    private final ProjectMapper projectMapper;
    private final CanvasAccessPolicy access;

    @Override
    public String getToolName() {
        return "list_my_projects";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "列出我的项目";
    }

    @Override
    public String getToolDescription() {
        return """
                列出用户能访问的所有项目。
                返回项目列表，包含每个项目的ID、名称、描述、封面等基本信息。
                当需要用户选择具体项目进行操作时，先调用此工具获取项目列表。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            List<Long> ids = access.readableProjects(context.requireMutationContext().actor());
            List<Project> projects = ids.isEmpty() ? List.of() : projectMapper.selectList(
                    new LambdaQueryWrapper<Project>()
                            .in(Project::getId, ids).orderByDesc(Project::getCreateTime));

            JSONArray projectList = new JSONArray();
            for (Project project : projects) {
                projectList.add(JSONUtil.createObj()
                        .set("projectId", project.getId())
                        .set("version", project.getVersion())
                        .set("name", project.getName())
                        .set("description", project.getDescription() != null && project.getDescription().length() > 100
                                ? project.getDescription().substring(0, 100) + "..."
                                : project.getDescription())
                        .set("coverUrl", project.getCoverUrl())
                        .set("status", project.getStatus()));
            }

            return JSONUtil.createObj()
                    .set("count", projects.size())
                    .set("projects", projectList)
                    .toString();
        } catch (Exception e) {
            log.error("列出用户项目失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }
}
