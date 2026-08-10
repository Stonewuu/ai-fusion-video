package com.stonewu.fusion.service.ai.tool.storyboard;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/** 通用更新分镜场次工具。 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateStoryboardSceneToolExecutor implements ToolExecutor {

    private final StoryboardService storyboardService;
    private final ProjectService projectService;

    @Override
    public String getToolName() {
        return "update_storyboard_scene";
    }

    @Override
    public String getDisplayName() {
        return "更新分镜场次";
    }

    @Override
    public String getToolDescription() {
        return "更新分镜场次的编号、标题、地点、时间、内外景、排序或状态。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "storyboardSceneId": {"type": "integer", "description": "分镜场次ID"},
                        "sceneNumber": {"type": "string", "description": "场次编号"},
                        "sceneHeading": {"type": "string", "description": "场次标题"},
                        "location": {"type": "string", "description": "地点"},
                        "timeOfDay": {"type": "string", "description": "时间段"},
                        "intExt": {"type": "string", "description": "内景或外景"},
                        "sortOrder": {"type": "integer", "description": "排序值"},
                        "status": {"type": "integer", "description": "状态"}
                    },
                    "required": ["storyboardSceneId"],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long sceneId = params.getLong("storyboardSceneId");
            if (sceneId == null) return error("缺少 storyboardSceneId");
            if (!hasUpdate(params)) return error("至少提供一个需要更新的字段");

            StoryboardScene existing = storyboardService.getSceneById(sceneId);
            Storyboard storyboard = storyboardService.getById(existing.getStoryboardId());
            if (!projectService.canAccessProject(storyboard.getProjectId(), context.getUserId())) {
                return error("无权更新该分镜场次");
            }

            StoryboardScene patch = new StoryboardScene();
            patch.setId(sceneId);
            patch.setSortOrder(null);
            patch.setStatus(null);
            if (params.containsKey("sceneNumber")) patch.setSceneNumber(params.getStr("sceneNumber"));
            if (params.containsKey("sceneHeading")) patch.setSceneHeading(params.getStr("sceneHeading"));
            if (params.containsKey("location")) patch.setLocation(params.getStr("location"));
            if (params.containsKey("timeOfDay")) patch.setTimeOfDay(params.getStr("timeOfDay"));
            if (params.containsKey("intExt")) patch.setIntExt(params.getStr("intExt"));
            if (params.containsKey("sortOrder")) patch.setSortOrder(params.getInt("sortOrder"));
            if (params.containsKey("status")) patch.setStatus(params.getInt("status"));

            StoryboardScene saved = storyboardService.updateScene(patch);
            return JSONUtil.createObj().set("status", "success").set("storyboardScene", saved).toString();
        } catch (Exception e) {
            log.error("更新分镜场次失败", e);
            return error("更新分镜场次失败: " + e.getMessage());
        }
    }

    private boolean hasUpdate(JSONObject params) {
        return params.containsKey("sceneNumber")
                || params.containsKey("sceneHeading")
                || params.containsKey("location")
                || params.containsKey("timeOfDay")
                || params.containsKey("intExt")
                || params.containsKey("sortOrder")
                || params.containsKey("status");
    }

    private String error(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
