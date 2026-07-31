package com.stonewu.fusion.service.ai.tool.script;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.tool.ToolResourceAccessGuard;
import com.stonewu.fusion.service.script.ScriptService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 保存集记录工具（save_script_episode）
 * <p>
 * 创建/更新集记录（标题、概述、原文），自动计算 sort_order。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SaveScriptEpisodeToolExecutor implements ToolExecutor {

    private final ScriptService scriptService;
    private final ToolResourceAccessGuard accessGuard;

    @Override
    public String getToolName() {
        return "save_script_episode";
    }

    @Override
    public String getDisplayName() {
        return "保存集记录";
    }

    @Override
    public String getToolDescription() {
        return """
                创建或更新剧本的一集记录。如果该集数已存在则更新，不存在则创建。
                建议传入 sortOrder 排序值进行排序（默认设置为集号），若不传入则默认使用集号进行排序。
                返回值中包含 episode_version，请在后续调用 save_script_scene_items 时传入此值。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "scriptId": {
                            "type": "integer",
                            "description": "剧本ID"
                        },
                        "episodeNumber": {
                            "type": "integer",
                            "description": "集数编号（如 1, 2, 3）"
                        },
                        "title": {
                            "type": "string",
                            "description": "集标题（如 '第一集' 或 '第一集：夜叩门'）"
                        },
                        "synopsis": {
                            "type": "string",
                            "description": "本集剧情概述（100-200字）"
                        },
                        "rawContent": {
                            "type": "string",
                            "description": "本集原始剧本文本"
                        },
                        "sourceType": {
                            "type": "integer",
                            "description": "内容来源：0-手动 1-AI创作 2-文本解析"
                        },
                        "sortOrder": {
                            "type": "integer",
                            "description": "剧本分集排序值。默认应设为与该集的 episodeNumber（集号）相同的值（例如第1集传 1，第2集传 2）。"
                        }
                    },
                    "required": ["scriptId", "episodeNumber", "title"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long scriptId = params.getLong("scriptId");
            Integer episodeNumber = params.getInt("episodeNumber");
            String title = params.getStr("title");
            String synopsis = params.getStr("synopsis");
            String rawContent = params.getStr("rawContent");
            Integer sourceType = params.getInt("sourceType");
            Integer sortOrder = params.getInt("sortOrder");

            if (scriptId == null || episodeNumber == null || title == null || title.isBlank()) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "缺少必要参数: scriptId, episodeNumber, title").toString();
            }

            accessGuard.requireScript(scriptId, context.getUserId());
            ScriptEpisode episode = scriptService.saveEpisode(scriptId, episodeNumber, title,
                    synopsis, rawContent, sourceType, sortOrder);

            JSONObject resultObj = JSONUtil.createObj()
                    .set("scriptEpisodeId", episode.getId())
                    .set("episodeNumber", episodeNumber)
                    .set("title", title)
                    .set("episode_version", episode.getVersion())
                    .set("message", String.format("第%d集 \"%s\" 保存成功", episodeNumber, title));

            return resultObj.toString();
        } catch (Exception e) {
            log.error("保存集记录失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "保存失败: " + e.getMessage()).toString();
        }
    }
}
