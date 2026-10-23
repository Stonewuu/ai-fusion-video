package com.stonewu.fusion.service.ai.tool.script;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.ToolPermissionRisk;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * 场次管理工具（manage_script_scenes）
 * <p>
 * 新增或删除场次（局部操作，不同于 save_script_scene_items 的整集替换）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ManageScriptSceneItemsToolExecutor implements ToolExecutor {

    private final ScriptToolCommands commands;


    @Override
    public String getToolName() {
        return "manage_script_scenes";
    }

    @Override
    public String getDisplayName() {
        return "管理剧本场次";
    }

    @Override
    public String getToolDescription() {
        return """
                新增或删除剧本场次（局部操作）。
                - add：在指定集中添加新剧本场次
                - delete：删除指定剧本场次
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "episode_version": { "type": "integer", "minimum": 0, "description": "刚刚查询的分集版本，保护整个场次集合" },
                        "action": {
                            "type": "string",
                            "enum": ["add", "delete"],
                            "description": "操作类型：add-新增 delete-删除"
                        },
                        "scriptEpisodeId": {
                            "type": "integer",
                            "description": "集ID（新增和删除均必填；删除对象必须属于同一集）"
                        },
                        "scriptSceneItemIds": {
                            "type": "array",
                            "items": { "type": "integer" },
                            "description": "要删除的场次ID列表（delete 操作必填）"
                        },
                        "scenes": {
                            "type": "array",
                            "description": "要新增的场次列表（add 操作必填）",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "scene_heading": { "type": "string" },
                                    "location": { "type": "string" },
                                    "time_of_day": { "type": "string" },
                                    "int_ext": { "type": "string" },
                                    "scene_description": { "type": "string" },
                                    "dialogues": {
                                        "type": "array",
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
                                "required": ["scene_heading"]
                            }
                        }
                    },
                    "required": ["action", "scriptEpisodeId", "episode_version"]
                }
                """;
    }

    @Override
    public ToolPermissionRisk getPermissionRisk(Map<String, Object> toolInput) {
        return "delete".equals(toolInput.get("action"))
                ? ToolPermissionRisk.HIGH_RISK
                : ToolPermissionRisk.EDIT;
    }

    @Override
    public boolean mayRequireHighRiskApproval() {
        return true;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return commands.manageScenes(toolInput, context);
    }
}
