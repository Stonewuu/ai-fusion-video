package com.stonewu.fusion.service.ai.tool.script;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 查询剧本工具（get_script）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ScriptQueryToolExecutor implements ToolExecutor {

    private final ScriptToolQueries queries;


    @Override
    public String getToolName() {
        return "get_script";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询剧本";
    }

    @Override
    public String getToolDescription() {
        return "查询剧本的基本信息（不含分集详情，如需分集请使用 get_script_structure）。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "scriptId": { "type": "integer", "description": "剧本ID" }
                    },
                    "required": ["scriptId"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return queries.execute(getToolName(), toolInput, context);
    }
}
