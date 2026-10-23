package com.stonewu.fusion.service.ai.tool.script;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 更新单个场次工具（update_script_scene）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateScriptSceneItemToolExecutor implements ToolExecutor {

    private final ScriptToolCommands commands;


    @Override
    public String getToolName() {
        return "update_script_scene";
    }

    @Override
    public String getDisplayName() {
        return "更新剧本场次";
    }

    @Override
    public String getToolDescription() {
        return """
                更新单个剧本场次的内容。可以修改场景标头、描述、对白等字段。
                仅传入需要修改的字段，未传入的字段保持不变。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "expectedVersion": { "type": "integer", "minimum": 0, "description": "刚刚读取的对象版本；创建分集传剧本版本，更新分集传分集版本" },
                        "scriptSceneItemId": { "type": "integer", "description": "剧本场次ID" },
                        "sceneHeading": { "type": "string", "description": "场景标头" },
                        "location": { "type": "string", "description": "场景地点" },
                        "timeOfDay": { "type": "string", "description": "时间" },
                        "intExt": { "type": "string", "description": "内外景" },
                        "characters": { "type": "array", "items": { "type": "string" }, "description": "出场角色名列表" },
                        "sceneDescription": { "type": "string", "description": "场景描述" },
                        "dialogues": {
                            "type": "array",
                            "description": "对白、动作和环境描写数组",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": { "type": "integer", "description": "1-对白 2-动作 3-VO 4-镜头指令 5-环境描写" },
                                    "character_name": { "type": "string" },
                                    "character_asset_id": { "type": "integer" },
                                    "parenthetical": { "type": "string" },
                                    "content": { "type": "string" }
                                },
                                "required": ["type", "content"]
                            }
                        }
                    },
                    "required": ["expectedVersion", "scriptSceneItemId"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return commands.patch(CanvasValueType.SCRIPT_SCENE, "scriptSceneItemId", toolInput, context);
    }
}
