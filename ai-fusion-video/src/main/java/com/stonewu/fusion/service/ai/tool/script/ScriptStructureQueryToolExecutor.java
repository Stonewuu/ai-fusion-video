package com.stonewu.fusion.service.ai.tool.script;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 剧本结构查询工具（get_script_structure）
 * <p>
 * 支持三级粒度返回：outline / summary / full
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ScriptStructureQueryToolExecutor implements ToolExecutor {

    private final ScriptToolQueries queries;


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
        return queries.execute(getToolName(), toolInput, context);
    }
}
