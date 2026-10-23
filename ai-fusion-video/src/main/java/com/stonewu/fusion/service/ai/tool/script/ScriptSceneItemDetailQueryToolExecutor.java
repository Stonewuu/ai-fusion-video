package com.stonewu.fusion.service.ai.tool.script;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 查询剧本单场次完整内容工具（get_script_scene）
 * <p>
 * 返回场次的完整信息：标头、环境描写、出场角色、完整对白/动作列表。
 * 用于分镜转换和子 Agent 场次解析时获取场次的全部细节。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ScriptSceneItemDetailQueryToolExecutor implements ToolExecutor {

    private final ScriptToolQueries queries;


    @Override
    public String getToolName() {
        return "get_script_scene";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询剧本场次详情";
    }

    @Override
    public String getToolDescription() {
        return """
                查询剧本中单个场次的完整内容，包含场景标头、环境描写、完整对白/动作列表、出场角色信息。
                用于分镜转换时获取场次的全部细节。""";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "scriptSceneItemId": {
                            "type": "integer",
                            "description": "剧本场次ID"
                        }
                    },
                    "required": ["scriptSceneItemId"]
                }""";
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return queries.execute(getToolName(), toolInput, context);
    }
}
