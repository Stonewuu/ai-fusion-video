package com.stonewu.fusion.service.ai.tool.script;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 更新剧本工具（update_script）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateScriptToolExecutor implements ToolExecutor {

    private final ScriptToolCommands commands;


    @Override
    public String getToolName() {
        return "update_script";
    }

    @Override
    public String getDisplayName() {
        return "更新剧本正文";
    }

    @Override
    public String getToolDescription() {
        return "仅更新剧本正文内容。故事梗概、人物表和类型请使用 update_script_info；顶层剧本名称固定跟随项目名称。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "expectedVersion": { "type": "integer", "minimum": 0, "description": "刚刚读取的对象版本；创建分集传剧本版本，更新分集传分集版本" },
                        "scriptId": { "type": "integer", "description": "剧本ID" },
                        "content": { "type": "string", "description": "剧本正文内容" }
                    },
                    "required": ["expectedVersion", "scriptId", "content"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return commands.patch(CanvasValueType.SCRIPT, "scriptId", toolInput, context);
    }
}
