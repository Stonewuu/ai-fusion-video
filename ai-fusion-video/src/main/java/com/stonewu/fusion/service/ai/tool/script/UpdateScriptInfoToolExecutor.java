package com.stonewu.fusion.service.ai.tool.script;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 更新剧本信息工具（update_script_info）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateScriptInfoToolExecutor implements ToolExecutor {

    private final ScriptToolCommands commands;


    @Override
    public String getToolName() {
        return "update_script_info";
    }

    @Override
    public String getDisplayName() {
        return "更新剧本信息";
    }

    @Override
    public String getToolDescription() {
        return """
                更新剧本的总体信息，包括故事梗概(storySynopsis)和人物表(charactersJson)等字段。
                scriptId 必填，各字段均为可选，仅传入需要更新的字段。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "expectedVersion": { "type": "integer", "minimum": 0, "description": "刚刚读取的对象版本；创建分集传剧本版本，更新分集传分集版本" },
                        "scriptId": { "type": "integer", "description": "剧本ID（必填）" },
                        "storySynopsis": { "type": "string", "description": "全剧故事梗概（200-500字概述）" },
                        "charactersJson": { "type": "array", "description": "人物表JSON快照", "items": { "type": "object", "properties": { "name": { "type": "string" }, "assetId": { "type": "integer" }, "description": { "type": "string" }, "importance": { "type": "string" } }, "required": ["name"] } },
                        "genre": { "type": "string", "description": "类型/风格" }
                    },
                    "required": ["expectedVersion", "scriptId"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return commands.patch(CanvasValueType.SCRIPT, "scriptId", toolInput, context);
    }
}
