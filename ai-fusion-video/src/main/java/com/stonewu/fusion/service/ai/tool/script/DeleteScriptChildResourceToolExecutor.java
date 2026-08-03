package com.stonewu.fusion.service.ai.tool.script;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.script.ScriptService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/** 删除剧本模块资源工具。 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DeleteScriptChildResourceToolExecutor implements ToolExecutor {

    private static final String RESOURCE_EPISODE = "episode";
    private static final String RESOURCE_SCENE = "scene";

    private final ScriptService scriptService;
    private final ProjectService projectService;

    @Override
    public String getToolName() {
        return "delete_script_child_resource";
    }

    @Override
    public String getDisplayName() {
        return "删除剧本子资源";
    }

    @Override
    public String getToolDescription() {
        return "删除剧本分集或场次。顶层剧本由项目生命周期维护，不允许删除。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "resourceType": {
                            "type": "string",
                            "enum": ["episode", "scene"],
                            "description": "要删除的资源类型"
                        },
                        "resourceId": {"type": "integer", "description": "要删除的资源ID"}
                    },
                    "required": ["resourceType", "resourceId"],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            String resourceType = params.getStr("resourceType");
            Long resourceId = params.getLong("resourceId");
            if (resourceType == null || resourceId == null) {
                return error("缺少 resourceType 或 resourceId");
            }

            Long scriptId = resolveScriptId(resourceType, resourceId);
            Script script = scriptService.getById(scriptId);
            if (!projectService.canAccessProject(script.getProjectId(), context.getUserId())) {
                return error("无权删除该剧本资源");
            }

            switch (resourceType) {
                case RESOURCE_EPISODE -> scriptService.deleteEpisode(resourceId);
                case RESOURCE_SCENE -> scriptService.deleteScene(resourceId);
                default -> {
                    return error("resourceType 仅支持 episode 或 scene");
                }
            }
            return JSONUtil.createObj()
                    .set("status", "success")
                    .set("deletedResourceType", resourceType)
                    .set("deletedResourceId", resourceId)
                    .toString();
        } catch (Exception e) {
            log.error("删除剧本资源失败", e);
            return error("删除剧本资源失败: " + e.getMessage());
        }
    }

    private Long resolveScriptId(String resourceType, Long resourceId) {
        return switch (resourceType) {
            case RESOURCE_EPISODE -> {
                ScriptEpisode episode = scriptService.getEpisodeById(resourceId);
                yield episode.getScriptId();
            }
            case RESOURCE_SCENE -> {
                ScriptSceneItem scene = scriptService.getSceneById(resourceId);
                if (scene.getScriptId() != null) {
                    yield scene.getScriptId();
                }
                yield scriptService.getEpisodeById(scene.getEpisodeId()).getScriptId();
            }
            default -> throw new IllegalArgumentException("resourceType 仅支持 episode 或 scene");
        };
    }

    private String error(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
