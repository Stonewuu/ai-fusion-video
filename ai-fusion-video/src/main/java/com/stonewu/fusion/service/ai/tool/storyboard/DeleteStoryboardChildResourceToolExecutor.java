package com.stonewu.fusion.service.ai.tool.storyboard;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/** 删除分镜模块资源工具。 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DeleteStoryboardChildResourceToolExecutor implements ToolExecutor {

    private static final String RESOURCE_EPISODE = "episode";
    private static final String RESOURCE_SCENE = "scene";
    private static final String RESOURCE_ITEM = "item";

    private final StoryboardService storyboardService;
    private final ProjectService projectService;

    @Override
    public String getToolName() {
        return "delete_storyboard_child_resource";
    }

    @Override
    public String getDisplayName() {
        return "删除分镜子资源";
    }

    @Override
    public String getToolDescription() {
        return "删除分镜集、场次或镜头。顶层分镜由项目生命周期维护，不允许删除。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "resourceType": {
                            "type": "string",
                            "enum": ["episode", "scene", "item"],
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

            Long storyboardId = resolveStoryboardId(resourceType, resourceId);
            Storyboard storyboard = storyboardService.getById(storyboardId);
            if (!projectService.canAccessProject(storyboard.getProjectId(), context.getUserId())) {
                return error("无权删除该分镜资源");
            }

            switch (resourceType) {
                case RESOURCE_EPISODE -> storyboardService.deleteEpisode(resourceId);
                case RESOURCE_SCENE -> storyboardService.deleteScene(resourceId);
                case RESOURCE_ITEM -> storyboardService.deleteItem(resourceId);
                default -> {
                    return error("resourceType 仅支持 episode、scene 或 item");
                }
            }
            return JSONUtil.createObj()
                    .set("status", "success")
                    .set("deletedResourceType", resourceType)
                    .set("deletedResourceId", resourceId)
                    .toString();
        } catch (Exception e) {
            log.error("删除分镜资源失败", e);
            return error("删除分镜资源失败: " + e.getMessage());
        }
    }

    private Long resolveStoryboardId(String resourceType, Long resourceId) {
        return switch (resourceType) {
            case RESOURCE_EPISODE -> {
                StoryboardEpisode episode = storyboardService.getEpisodeById(resourceId);
                yield episode.getStoryboardId();
            }
            case RESOURCE_SCENE -> {
                StoryboardScene scene = storyboardService.getSceneById(resourceId);
                yield scene.getStoryboardId();
            }
            case RESOURCE_ITEM -> {
                StoryboardItem item = storyboardService.getItemById(resourceId);
                if (item.getStoryboardId() != null) {
                    yield item.getStoryboardId();
                }
                yield storyboardService.getSceneById(item.getStoryboardSceneId()).getStoryboardId();
            }
            default -> throw new IllegalArgumentException(
                    "resourceType 仅支持 episode、scene 或 item");
        };
    }

    private String error(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
