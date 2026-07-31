package com.stonewu.fusion.service.ai.tool.script;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.script.ScriptService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

import org.springframework.stereotype.Component;

/**
 * 查询项目剧本元数据工具（get_project_script）
 * <p>
 * 一个项目只有一个剧本，返回该剧本的基础信息和分集概览。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class GetProjectScriptToolExecutor implements ToolExecutor {

    private final ScriptService scriptService;
    private final ProjectService projectService;

    @Override
    public String getToolName() {
        return "get_project_script";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询项目剧本信息";
    }

    @Override
    public String getToolDescription() {
        return """
                查询指定项目的剧本元数据（一个项目只有一个剧本）。
                返回剧本ID、标题、总集数、解析状态、故事梗概、原文内容以及分集概览。
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
                            "description": "项目ID（必填）"
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
            if (projectId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 projectId").toString();
            }

            Long userId = context.getUserId();

            if (!projectService.canAccessProject(projectId, userId)) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "无权访问该项目").toString();
            }

            Script script = scriptService.getByProjectId(projectId);
            if (script == null) {
                return JSONUtil.createObj()
                        .set("projectId", projectId)
                        .set("status", "empty")
                        .set("message", "该项目下尚无剧本").toString();
            }
            JSONObject result = JSONUtil.createObj()
                    .set("projectId", projectId)
                    .set("scriptId", script.getId())
                    .set("title", script.getTitle())
                    .set("totalEpisodes", script.getTotalEpisodes())
                    .set("parsingStatus", script.getParsingStatus())
                    .set("storySynopsis", script.getStorySynopsis())
                    .set("rawContent", script.getRawContent())
                    .set("genre", script.getGenre());

            // 附带分集概览
            List<ScriptEpisode> episodes = scriptService.listEpisodes(script.getId());
            JSONArray episodeList = new JSONArray();
            for (ScriptEpisode ep : episodes) {
                episodeList.add(JSONUtil.createObj()
                        .set("scriptEpisodeId", ep.getId())
                        .set("episodeNumber", ep.getEpisodeNumber())
                        .set("title", ep.getTitle())
                        .set("totalScenes", ep.getTotalScenes()));
            }
            result.set("episodes", episodeList);

            return result.toString();
        } catch (Exception e) {
            log.error("查询项目剧本信息失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }
}
