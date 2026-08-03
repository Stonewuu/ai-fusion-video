package com.stonewu.fusion.service.ai.tool.script;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.tool.ToolResourceAccessGuard;
import com.stonewu.fusion.service.script.ScriptService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 剧本结构查询工具（get_script_structure）
 * <p>
 * 支持三级粒度返回：outline / summary / full
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ScriptStructureQueryToolExecutor implements ToolExecutor {

    private final ScriptService scriptService;
    private final ToolResourceAccessGuard accessGuard;

    @Override
    public String getToolName() {
        return "get_script_structure";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询剧本结构";
    }

    @Override
    public String getToolDescription() {
        return """
                查询剧本的整体结构，支持不同详细程度：
                - outline（默认）：集列表和场次标题，快速了解骨架
                - summary：集概述 + 场次概述（含场景描述和角色列表，不含对白），适合了解剧情上下文
                - full：完整信息含所有对白，数据量大，慎用
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "projectId": {
                            "type": "integer",
                            "description": "项目ID"
                        },
                        "detailLevel": {
                            "type": "string",
                            "enum": ["outline", "summary", "full"],
                            "description": "返回详细程度。outline=集列表+场次标题（默认）；summary=集概述+场次概述（含描述和角色）；full=完整信息含对白"
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
            String detailLevel = params.getStr("detailLevel", "outline");

            if (projectId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 projectId").toString();
            }
            accessGuard.requireProject(projectId, context.getUserId());

            Script script = scriptService.getByProjectId(projectId);
            if (script == null) {
                return JSONUtil.createObj().set("status", "success").set("result", "该项目尚无剧本").toString();
            }

            List<ScriptEpisode> episodes = scriptService.listEpisodes(script.getId());

            JSONObject scriptInfo = JSONUtil.createObj()
                    .set("scriptId", script.getId())
                    .set("title", script.getTitle())
                    .set("totalEpisodes", script.getTotalEpisodes())
                    .set("parsingStatus", script.getParsingStatus());

            List<JSONObject> episodeList = episodes.stream().map(ep -> {
                JSONObject epObj = JSONUtil.createObj()
                        .set("scriptEpisodeId", ep.getId())
                        .set("episodeNumber", ep.getEpisodeNumber())
                        .set("title", ep.getTitle())
                        .set("synopsis", ep.getSynopsis())
                        .set("totalScenes", ep.getTotalScenes())
                        .set("version", ep.getVersion());

                List<ScriptSceneItem> items = scriptService.listScenesByEpisode(ep.getId());

                switch (detailLevel) {
                    case "summary" -> {
                        List<JSONObject> sceneSummaries = items.stream()
                                .map(item -> JSONUtil.createObj()
                                        .set("scriptSceneItemId", item.getId())
                                        .set("sceneNumber", item.getSceneNumber())
                                        .set("sceneHeading", item.getSceneHeading())
                                        .set("sceneDescription", item.getSceneDescription())
                                        .set("characters", item.getCharacters()))
                                .collect(Collectors.toList());
                        epObj.set("scenes", sceneSummaries);
                    }
                    case "full" -> {
                        List<JSONObject> scenesFull = items.stream()
                                .map(item -> JSONUtil.createObj()
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
                                        .set("dialogues", item.getDialogues()))
                                .collect(Collectors.toList());
                        epObj.set("scenes", scenesFull);
                    }
                    default -> {
                        List<JSONObject> sceneOutlines = items.stream()
                                .map(item -> JSONUtil.createObj()
                                        .set("scriptSceneItemId", item.getId())
                                        .set("sceneNumber", item.getSceneNumber())
                                        .set("sceneHeading", item.getSceneHeading()))
                                .collect(Collectors.toList());
                        epObj.set("scenes", sceneOutlines);
                    }
                }
                return epObj;
            }).collect(Collectors.toList());

            scriptInfo.set("episodes", episodeList);
            return scriptInfo.toString();
        } catch (Exception e) {
            log.error("查询剧本结构失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }
}
