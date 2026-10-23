package com.stonewu.fusion.service.ai.tool.script;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;


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

    private final ScriptToolQueries queries;


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
        return queries.execute(getToolName(), toolInput, context);
    }
}
