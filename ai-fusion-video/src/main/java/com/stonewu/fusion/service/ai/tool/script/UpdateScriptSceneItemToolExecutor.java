package com.stonewu.fusion.service.ai.tool.script;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.tool.ToolResourceAccessGuard;
import com.stonewu.fusion.service.script.ScriptService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 更新单个场次工具（update_script_scene）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateScriptSceneItemToolExecutor implements ToolExecutor {

    private final ScriptService scriptService;
    private final ToolResourceAccessGuard accessGuard;

    @Override
    public String getToolName() {
        return "update_script_scene";
    }

    @Override
    public String getDisplayName() {
        return "更新剧本场次";
    }

    @Override
    public String getToolDescription() {
        return """
                更新单个剧本场次的内容。可以修改场景标头、描述、对白等字段。
                仅传入需要修改的字段，未传入的字段保持不变。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "scriptSceneItemId": { "type": "integer", "description": "剧本场次ID" },
                        "sceneHeading": { "type": "string", "description": "场景标头" },
                        "location": { "type": "string", "description": "场景地点" },
                        "timeOfDay": { "type": "string", "description": "时间" },
                        "intExt": { "type": "string", "description": "内外景" },
                        "characters": { "type": "array", "items": { "type": "string" }, "description": "出场角色名列表" },
                        "sceneDescription": { "type": "string", "description": "场景描述" },
                        "dialogues": {
                            "type": "array",
                            "description": "对白、动作和环境描写数组",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": { "type": "integer", "description": "1-对白 2-动作 3-VO 4-镜头指令 5-环境描写" },
                                    "character_name": { "type": "string" },
                                    "character_asset_id": { "type": "integer" },
                                    "parenthetical": { "type": "string" },
                                    "content": { "type": "string" }
                                },
                                "required": ["type", "content"]
                            }
                        }
                    },
                    "required": ["scriptSceneItemId"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long scriptSceneItemId = params.getLong("scriptSceneItemId");
            if (scriptSceneItemId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 scriptSceneItemId").toString();
            }

            ScriptSceneItem item = accessGuard.requireScriptScene(scriptSceneItemId, context.getUserId());
            if (params.containsKey("sceneHeading")) {
                item.setSceneHeading(params.getStr("sceneHeading"));
            }
            if (params.containsKey("location")) {
                item.setLocation(params.getStr("location"));
            }
            if (params.containsKey("timeOfDay")) {
                item.setTimeOfDay(params.getStr("timeOfDay"));
            }
            if (params.containsKey("intExt")) {
                item.setIntExt(params.getStr("intExt"));
            }
            if (params.containsKey("characters")) {
                item.setCharacters(params.getJSONArray("characters").toString());
            }
            if (params.containsKey("sceneDescription")) {
                item.setSceneDescription(params.getStr("sceneDescription"));
            }
            if (params.containsKey("dialogues")) {
                item.setDialogues(params.getJSONArray("dialogues").toString());
            }

            scriptService.updateScene(item);
            return JSONUtil.createObj()
                    .set("scriptSceneItemId", scriptSceneItemId)
                    .set("message", "场次更新成功").toString();
        } catch (Exception e) {
            log.error("更新场次失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "更新失败: " + e.getMessage()).toString();
        }
    }
}
