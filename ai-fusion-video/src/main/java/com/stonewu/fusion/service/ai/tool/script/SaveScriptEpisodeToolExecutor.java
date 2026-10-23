package com.stonewu.fusion.service.ai.tool.script;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
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

    private final ScriptToolCommands commands;


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
                创建或更新剧本的一集记录。更新必须传实际 scriptEpisodeId 和该分集的 expectedVersion；创建不传 scriptEpisodeId，expectedVersion 传剧本版本。已存在的集号不能用于隐式覆盖。
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
                        "scriptEpisodeId": { "type": "integer", "description": "更新时必填实际分集ID；创建时不传，已有集号不会隐式覆盖" },
                        "expectedVersion": { "type": "integer", "minimum": 0, "description": "刚刚读取的对象版本；创建分集传剧本版本，更新分集传分集版本" },
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
                    "required": ["expectedVersion", "scriptId", "episodeNumber", "title"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return commands.saveEpisode(toolInput, context);
    }
}
