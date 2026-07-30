package com.stonewu.fusion.service.ai.tool;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.ToolPermissionRisk;
import com.stonewu.fusion.service.script.ScriptService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 场次管理工具（manage_script_scenes）
 * <p>
 * 新增或删除场次（局部操作，不同于 save_script_scene_items 的整集替换）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ManageScriptSceneItemsToolExecutor implements ToolExecutor {

    private final ScriptService scriptService;

    @Override
    public String getToolName() {
        return "manage_script_scenes";
    }

    @Override
    public String getDisplayName() {
        return "管理剧本场次";
    }

    @Override
    public String getToolDescription() {
        return """
                新增或删除剧本场次（局部操作）。
                - add：在指定集中添加新剧本场次
                - delete：删除指定剧本场次
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "delete"],
                            "description": "操作类型：add-新增 delete-删除"
                        },
                        "scriptEpisodeId": {
                            "type": "number",
                            "description": "集ID（add 操作必填）"
                        },
                        "scriptSceneItemIds": {
                            "type": "array",
                            "items": { "type": "number" },
                            "description": "要删除的场次ID列表（delete 操作必填）"
                        },
                        "scenes": {
                            "type": "array",
                            "description": "要新增的场次列表（add 操作必填）",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "scene_heading": { "type": "string" },
                                    "location": { "type": "string" },
                                    "time_of_day": { "type": "string" },
                                    "int_ext": { "type": "string" },
                                    "scene_description": { "type": "string" },
                                    "dialogues": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "type": { "type": "number", "description": "1-对白 2-动作 3-VO 4-镜头指令 5-环境描写" },
                                                "character_name": { "type": "string" },
                                                "character_asset_id": { "type": "number" },
                                                "parenthetical": { "type": "string" },
                                                "content": { "type": "string" }
                                            },
                                            "required": ["type", "content"]
                                        }
                                    }
                                },
                                "required": ["scene_heading"]
                            }
                        }
                    },
                    "required": ["action"]
                }
                """;
    }

    @Override
    public ToolPermissionRisk getPermissionRisk(Map<String, Object> toolInput) {
        return "delete".equals(toolInput.get("action"))
                ? ToolPermissionRisk.HIGH_RISK
                : ToolPermissionRisk.EDIT;
    }

    @Override
    public boolean mayRequireHighRiskApproval() {
        return true;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            String action = params.getStr("action");

            if ("delete".equals(action)) {
                JSONArray ids = params.getJSONArray("scriptSceneItemIds");
                if (ids == null || ids.isEmpty()) {
                    return JSONUtil.createObj().set("status", "error").set("message", "删除操作需提供 scriptSceneItemIds")
                            .toString();
                }
                List<Long> deletedIds = new ArrayList<>();
                for (int i = 0; i < ids.size(); i++) {
                    Long id = ids.getLong(i);
                    scriptService.deleteScene(id);
                    deletedIds.add(id);
                }
                return JSONUtil.createObj()
                        .set("action", "delete")
                        .set("deletedCount", deletedIds.size())
                        .set("message", String.format("成功删除 %d 个场次", deletedIds.size())).toString();

            } else if ("add".equals(action)) {
                Long scriptEpisodeId = params.getLong("scriptEpisodeId");
                if (scriptEpisodeId == null) {
                    return JSONUtil.createObj().set("status", "error").set("message", "新增操作需提供 scriptEpisodeId").toString();
                }
                JSONArray scenesArray = params.getJSONArray("scenes");
                if (scenesArray == null || scenesArray.isEmpty()) {
                    return JSONUtil.createObj().set("status", "error").set("message", "新增操作需提供 scenes").toString();
                }

                ScriptEpisode episode = scriptService.getEpisodeById(scriptEpisodeId);
                List<ScriptSceneItem> existing = scriptService.listScenesByEpisode(scriptEpisodeId);
                int nextOrder = existing.isEmpty() ? 0 : existing.get(existing.size() - 1).getSortOrder() + 1;

                List<Long> createdIds = new ArrayList<>();
                for (int i = 0; i < scenesArray.size(); i++) {
                    JSONObject sceneJson = scenesArray.getJSONObject(i);
                    ScriptSceneItem item = ScriptSceneItem.builder()
                            .episodeId(scriptEpisodeId)
                            .scriptId(episode.getScriptId())
                            .sceneNumber(String.format("%d-%d", episode.getEpisodeNumber(), nextOrder + i + 1))
                            .sceneHeading(sceneJson.getStr("scene_heading"))
                            .location(sceneJson.getStr("location"))
                            .timeOfDay(sceneJson.getStr("time_of_day"))
                            .intExt(sceneJson.getStr("int_ext"))
                            .sceneDescription(sceneJson.getStr("scene_description"))
                            .dialogues(
                                    sceneJson.containsKey("dialogues") ? sceneJson.getJSONArray("dialogues").toString()
                                            : null)
                            .sortOrder(nextOrder + i)
                            .status(1)
                            .build();
                    ScriptSceneItem saved = scriptService.createScene(item);
                    createdIds.add(saved.getId());
                }

                return JSONUtil.createObj()
                        .set("action", "add")
                        .set("createdCount", createdIds.size())
                        .set("createdIds", createdIds)
                        .set("message", String.format("成功新增 %d 个场次", createdIds.size())).toString();
            } else {
                return JSONUtil.createObj().set("status", "error").set("message", "无效的 action: " + action).toString();
            }
        } catch (Exception e) {
            log.error("管理场次失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "操作失败: " + e.getMessage()).toString();
        }
    }
}
