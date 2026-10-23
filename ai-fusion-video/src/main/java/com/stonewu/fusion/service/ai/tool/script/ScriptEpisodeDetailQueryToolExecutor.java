package com.stonewu.fusion.service.ai.tool.script;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 集详情查询工具（get_script_episode）
 * <p>
 * 支持三级粒度：summary / full / scenes_only
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ScriptEpisodeDetailQueryToolExecutor implements ToolExecutor {

    private final ScriptToolQueries queries;


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
        return queries.execute(getToolName(), toolInput, context);
    }
}
