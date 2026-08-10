package com.stonewu.fusion.service.ai.tool.script;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.tool.ToolResourceAccessGuard;
import com.stonewu.fusion.service.script.ScriptService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 集详情查询工具（get_script_episode）
 * <p>
 * 支持三级粒度：summary / full / scenes_only
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ScriptEpisodeDetailQueryToolExecutor implements ToolExecutor {

    private final ScriptService scriptService;
    private final ToolResourceAccessGuard accessGuard;

    @Override
    public String getToolName() {
        return "get_script_episode";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询集详情";
    }

    @Override
    public String getToolDescription() {
        return """
                查询某一集的内容，支持不同详细程度：
                - summary（默认）：返回集基本信息、episode_version 和场次概要（不含对白），适合了解轮廓和获取 version
                - full：返回完整内容含所有对白，数据量大
                - scenes_only：仅返回 scriptSceneItemIds 指定场次的完整对白，适合按需查看前后衔接
                返回值中包含 episode_version，调用 save_script_scene_items 时必须传入。
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
                            "description": "剧本集ID"
                        },
                        "detailLevel": {
                            "type": "string",
                            "enum": ["summary", "full", "scenes_only"],
                            "description": "返回详细程度。summary=概要含version和场次概述不含对白（默认）；full=完整含对白；scenes_only=仅返回scriptSceneItemIds指定场次的完整对白"
                        },
                        "scriptSceneItemIds": {
                            "type": "array",
                            "items": { "type": "integer" },
                            "description": "当 detailLevel=scenes_only 时必传，指定要查看完整对白的场次ID列表"
                        }
                    },
                    "required": ["scriptEpisodeId"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long scriptEpisodeId = params.getLong("scriptEpisodeId");
            if (scriptEpisodeId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少必要参数: scriptEpisodeId").toString();
            }

            String detailLevel = params.getStr("detailLevel", "summary");

            ScriptEpisode episode = accessGuard.requireScriptEpisode(scriptEpisodeId, context.getUserId());
            List<ScriptSceneItem> sceneItems = scriptService.listScenesByEpisode(scriptEpisodeId);

            JSONObject episodeObj = JSONUtil.createObj()
                    .set("scriptEpisodeId", episode.getId())
                    .set("scriptId", episode.getScriptId())
                    .set("episodeNumber", episode.getEpisodeNumber())
                    .set("title", episode.getTitle())
                    .set("synopsis", episode.getSynopsis())
                    .set("totalScenes", episode.getTotalScenes())
                    .set("episode_version", episode.getVersion());

            switch (detailLevel) {
                case "full" -> {
                    episodeObj.set("rawContent", episode.getRawContent());
                    List<JSONObject> scenes = sceneItems.stream().map(item -> JSONUtil.createObj()
                            .set("scriptSceneItemId", item.getId())
                            .set("sceneNumber", item.getSceneNumber())
                            .set("sceneHeading", item.getSceneHeading())
                            .set("location", item.getLocation())
                            .set("timeOfDay", item.getTimeOfDay())
                            .set("intExt", item.getIntExt())
                            .set("characters", item.getCharacters())
                            .set("characterAssetIds", item.getCharacterAssetIds())
                            .set("sceneAssetId", item.getSceneAssetId())
                            .set("sceneDescription", item.getSceneDescription())
                            .set("dialogues", item.getDialogues())).toList();
                    episodeObj.set("scenes", scenes);
                }
                case "scenes_only" -> {
                    Set<Long> targetIds = new HashSet<>();
                    if (params.containsKey("scriptSceneItemIds")) {
                        params.getJSONArray("scriptSceneItemIds").forEach(id -> targetIds.add(Long.valueOf(id.toString())));
                    }
                    if (targetIds.isEmpty()) {
                        return JSONUtil.createObj().set("status", "error")
                                .set("message", "detailLevel=scenes_only 时必须提供 scriptSceneItemIds 参数").toString();
                    }
                    List<JSONObject> scenes = sceneItems.stream()
                             .filter(item -> targetIds.contains(item.getId()))
                            .map(item -> JSONUtil.createObj()
                                    .set("scriptSceneItemId", item.getId())
                                    .set("sceneNumber", item.getSceneNumber())
                                    .set("sceneHeading", item.getSceneHeading())
                                    .set("characters", item.getCharacters())
                                    .set("sceneDescription", item.getSceneDescription())
                                    .set("dialogues", item.getDialogues()))
                            .toList();
                    episodeObj.set("scenes", scenes);
                }
                default -> {
                    List<JSONObject> scenes = sceneItems.stream().map(item -> JSONUtil.createObj()
                            .set("scriptSceneItemId", item.getId())
                            .set("sceneNumber", item.getSceneNumber())
                            .set("sceneHeading", item.getSceneHeading())
                            .set("characters", item.getCharacters())
                            .set("sceneDescription", item.getSceneDescription())).toList();
                    episodeObj.set("scenes", scenes);
                }
            }

            return episodeObj.toString();
        } catch (Exception e) {
            log.error("查询集详情失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }
}
