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
import org.springframework.util.StringUtils;

/** 创建或更新项目工具。 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SaveProjectToolExecutor implements ToolExecutor {

    private static final int OWNER_TYPE_PERSONAL = 1;

    private final ProjectService projectService;

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
        return "创建项目，或传入 projectId 更新当前用户可访问的项目。未传 projectId 时 name 必填。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "projectId": {"type": "integer", "description": "项目ID；不传表示创建"},
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
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long projectId = params.getLong("projectId");
            Project project;
            String operation;

            if (projectId == null) {
                String name = params.getStr("name");
                if (!StringUtils.hasText(name)) {
                    return error("创建项目时 name 必填");
                }
                project = new Project();
                project.setName(name);
                project.setOwnerType(OWNER_TYPE_PERSONAL);
                project.setOwnerId(context.getUserId());
                applyFields(project, params);
                project = projectService.create(project);
                operation = "created";
            } else {
                Project existing = projectService.getById(projectId);
                if (!projectService.canAccessProject(existing, context.getUserId())) {
                    return error("无权更新该项目");
                }
                Project patch = new Project();
                patch.setId(projectId);
                patch.setScope(null);
                patch.setStatus(null);
                applyFields(patch, params);
                projectService.update(patch);
                project = projectService.getById(projectId);
                operation = "updated";
            }

            return JSONUtil.createObj()
                    .set("status", "success")
                    .set("operation", operation)
                    .set("project", project)
                    .toString();
        } catch (Exception e) {
            log.error("保存项目失败", e);
            return error("保存项目失败: " + e.getMessage());
        }
    }

    private void applyFields(Project project, JSONObject params) {
        if (params.containsKey("name")) project.setName(params.getStr("name"));
        if (params.containsKey("description")) project.setDescription(params.getStr("description"));
        if (params.containsKey("coverUrl")) project.setCoverUrl(params.getStr("coverUrl"));
        if (params.containsKey("properties")) project.setProperties(params.getStr("properties"));
        if (params.containsKey("artStyle")) project.setArtStyle(params.getStr("artStyle"));
        if (params.containsKey("artStyleDescription")) {
            project.setArtStyleDescription(params.getStr("artStyleDescription"));
        }
        if (params.containsKey("artStyleImagePrompt")) {
            project.setArtStyleImagePrompt(params.getStr("artStyleImagePrompt"));
        }
        if (params.containsKey("artStyleImageUrl")) {
            project.setArtStyleImageUrl(params.getStr("artStyleImageUrl"));
        }
        if (params.containsKey("status")) project.setStatus(params.getInt("status"));
    }

    private String error(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
