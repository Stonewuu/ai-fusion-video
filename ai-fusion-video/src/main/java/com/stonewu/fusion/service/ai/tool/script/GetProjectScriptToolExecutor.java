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
                分页查询剧本与分集梗概，不返回原文或场次全文。默认每页 30 集，最多 50 集。
                hasMore=true 时使用 nextOffset 继续读取目录；用 get_script_episode 读取一集的场次，get_script_scene 读取具体场次。
                全量解析将分集 ID 交给单集解析子任务；分镜和视频流程直接使用已解析的场次与梗概。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "offset": {"type":"integer","minimum":0,"description":"目录起始位置；继续读取使用 nextOffset"},
                        "limit": {"type":"integer","minimum":1,"maximum":50,"description":"每页集数，默认 30"},
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
