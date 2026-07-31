package com.stonewu.fusion.service.ai.tool.script;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.ToolPermissionRisk;
import com.stonewu.fusion.service.ai.tool.ToolResourceAccessGuard;
import com.stonewu.fusion.service.script.ScriptService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 批量保存场次工具（save_script_scene_items）
 * <p>
 * 支持覆盖或追加写入场次数据。需传入 episode_version 做乐观锁校验。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SaveScriptSceneItemsToolExecutor implements ToolExecutor {

    private final ScriptService scriptService;
    private final ToolResourceAccessGuard accessGuard;

    @Override
    public String getToolName() {
        return "save_script_scene_items";
    }

    @Override
    public String getDisplayName() {
        return "保存场次数据";
    }

    @Override
    public String getToolDescription() {
        return """
                批量写入某集的场次和对白数据。调用前必须获取 episode_version（通过 save_script_episode 返回值或 get_script_episode 返回值获取）。
                工具内部按传入 scenes 数组顺序自动赋值 sort_order 和 scene_number，无需显式传入。
                默认模式（overwriteMode=false 或不传）为追加模式，不删除已有场次；仅当 overwriteMode=true 时，才会先清空该集所有旧场次再写入当前数据。

                ## 分批调用规则（必须遵守！）
                - 每次调用 scenes 数组最多传入 2 个场次
                - 如果一集有超过 2 个场次，必须分多次调用
                - 第一次调用传入前 1-2 个场次，overwriteMode 必须设为 true（会清空旧数据）
                - 第二次及之后每次传入后续 1-2 个场次，overwriteMode 不传或设为 false（追加模式）

                ## scene_heading 格式规则
                场景标头格式为：「内/外景 + 地点 + 时间」。示例：
                - "内景 张三家客厅 夜"
                - "外景 学校操场 日"
                - "内/外景 咖啡馆 黄昏"
                从标头中拆解出 location (地点)、time_of_day (时间)、int_ext (内/外景) 分别填入对应字段。

                ## dialogues 数组构建规则
                必须按剧本原文的严格时间顺序逐行拆解，每个元素对应一个 dialogue 条目。
                type 值：
                - 1：对白 — character_name 必填，content 填台词文本
                - 2：动作描写 — 原文中以 ▲ 开头的行，content 填动作描述
                - 3：画外音(V.O.) — character_name 填说话人，content 填画外音内容
                - 4：镜头指令 — 原文中【】包裹的内容，如【切】【闪回】
                - 5：环境描写 — 场景氛围、环境描述的文字

                **重要规则**：
                - 不要合并、跳过或省略任何行，严格按原文顺序逐行拆解
                - 对白（type=1）的 character_name 必须与 characters 列表中的角色名完全一致
                - parenthetical 填写括号注释，如"（低声）"、"（愤怒地）"
                - 若 character_asset_id 已知，请同时填入以建立角色关联
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "scriptEpisodeId": {
                            "type": "integer",
                            "description": "集ID"
                        },
                        "episode_version": {
                            "type": "integer",
                            "description": "集的版本号（从 get_script_episode 返回值获取，用于乐观锁校验）"
                        },
                        "scenes": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 2,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "scene_heading": { "type": "string", "description": "场景标头" },
                                    "location": { "type": "string", "description": "场景地点" },
                                    "time_of_day": { "type": "string", "description": "时间" },
                                    "int_ext": { "type": "string", "description": "内外景" },
                                    "characters": { "type": "array", "items": { "type": "string" }, "description": "出场角色名列表" },
                                    "character_asset_ids": { "type": "array", "items": { "type": "integer" }, "description": "角色资产ID列表" },
                                    "scene_asset_id": { "type": "integer", "description": "场景资产ID" },
                                    "prop_asset_ids": { "type": "array", "items": { "type": "integer" }, "description": "道具资产ID列表" },
                                    "scene_description": { "type": "string", "description": "场景氛围概述" },
                                    "dialogues": {
                                        "type": "array",
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
                                "required": ["scene_heading"]
                            },
                            "description": "场次列表（按顺序排列，每次最多传入2个场次）"
                        },
                        "overwriteMode": {
                            "type": "boolean",
                            "description": "覆盖模式。true=先删除该集旧场次再写入当前 scenes；false或不传=追加到已有场次后。分批写入时通常仅第一次调用设为 true"
                        }
                    },
                    "required": ["scriptEpisodeId", "episode_version", "scenes"]
                }
                """;
    }

    @Override
    public ToolPermissionRisk getPermissionRisk(Map<String, Object> toolInput) {
        return Boolean.TRUE.equals(toolInput.get("overwriteMode"))
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
            Long scriptEpisodeId = params.getLong("scriptEpisodeId");
            Integer episodeVersion = params.getInt("episode_version");
            JSONArray scenesArray = params.getJSONArray("scenes");
            boolean overwriteMode = Boolean.TRUE.equals(params.getBool("overwriteMode"));

            if (scriptEpisodeId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少必要参数: scriptEpisodeId").toString();
            }
            if (scenesArray == null || scenesArray.isEmpty() || scenesArray.size() > 2) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "scenes 每次必须传入 1-2 个场次").toString();
            }
            accessGuard.requireScriptEpisode(scriptEpisodeId, context.getUserId());

            List<ScriptSceneItem> sceneItems = new ArrayList<>();
            for (int i = 0; i < scenesArray.size(); i++) {
                JSONObject sceneJson = scenesArray.getJSONObject(i);
                ScriptSceneItem item = ScriptSceneItem.builder()
                            .sceneHeading(sceneJson.getStr("scene_heading"))
                            .location(sceneJson.getStr("location"))
                            .timeOfDay(sceneJson.getStr("time_of_day"))
                            .intExt(sceneJson.getStr("int_ext"))
                            .sceneDescription(sceneJson.getStr("scene_description"))
                            .sceneAssetId(sceneJson.getLong("scene_asset_id"))
                            .characters(sceneJson.containsKey("characters")
                                    ? sceneJson.getJSONArray("characters").toString()
                                    : null)
                            .characterAssetIds(sceneJson.containsKey("character_asset_ids")
                                    ? sceneJson.getJSONArray("character_asset_ids").toString()
                                    : null)
                            .propAssetIds(sceneJson.containsKey("prop_asset_ids")
                                    ? sceneJson.getJSONArray("prop_asset_ids").toString()
                                    : null)
                            .dialogues(
                                    sceneJson.containsKey("dialogues") ? sceneJson.getJSONArray("dialogues").toString()
                                            : null)
                            .status(1)
                        .build();
                sceneItems.add(item);
            }

            scriptService.batchSaveSceneItems(scriptEpisodeId, episodeVersion, sceneItems, overwriteMode);

            JSONObject resultObj = JSONUtil.createObj()
                    .set("scriptEpisodeId", scriptEpisodeId)
                    .set("sceneCount", sceneItems.size())
                    .set("overwriteMode", overwriteMode)
                    .set("message", String.format("成功%s写入 %d 个场次", overwriteMode ? "覆盖" : "追加", sceneItems.size()));

            return resultObj.toString();
        } catch (Exception e) {
            log.error("保存场次数据失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "保存失败: " + e.getMessage()).toString();
        }
    }
}
