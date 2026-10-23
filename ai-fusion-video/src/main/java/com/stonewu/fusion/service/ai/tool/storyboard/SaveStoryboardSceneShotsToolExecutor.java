package com.stonewu.fusion.service.ai.tool.storyboard;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 保存场次分镜工具（save_storyboard_scene_shots）
 * <p>
 * 创建一个分镜场次，并批量插入该场次下的所有镜头。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SaveStoryboardSceneShotsToolExecutor implements ToolExecutor {
    private final StoryboardToolCommands tools;


    @Override
    public String getToolName() {
        return "save_storyboard_scene_shots";
    }

    @Override
    public String getDisplayName() {
        return "保存场次分镜";
    }

    @Override
    public String getToolDescription() {
        return """
                创建一个分镜场次，并批量保存该场次下的所有镜头。
                可一次性保存一个场次的完整分镜数据。

                【参数说明】
                - storyboardId：所属分镜脚本ID（必填）
                - storyboardEpisodeId：所属分镜集ID（必填）
                - sceneNumber：场次编号，如 "1-1"（必填）
                - sceneHeading：场景标头，如 "内景 客厅 夜"
                - location：场景地点
                - timeOfDay：时间段（日/夜/黄昏/清晨）
                - intExt：内外景标识（内景/外景/内外景）
                - shots：镜头数组（必填），每个镜头含 content、shotType、duration、dialogue 等
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "storyboardId": {
                            "type": "integer",
                            "description": "所属分镜脚本ID（必填）"
                        },
                        "storyboardEpisodeId": {
                            "type": "integer",
                            "description": "所属分镜集ID（必填）"
                        },
                        "sceneNumber": {
                            "type": "string",
                            "description": "场次编号，如 1-1（必填）"
                        },
                        "sceneHeading": {
                            "type": "string",
                            "description": "场景标头，如 内景 客厅 夜"
                        },
                        "location": {
                            "type": "string",
                            "description": "场景地点"
                        },
                        "timeOfDay": {
                            "type": "string",
                            "description": "时间段（日/夜/黄昏/清晨）"
                        },
                        "intExt": {
                            "type": "string",
                            "description": "内外景标识（内景/外景/内外景）"
                        },
                        "sortOrder": {
                            "type": "integer",
                            "description": "明确插入位置，从 0 开始；缺省时追加到分集末尾"
                        },
                        "shots": {
                            "type": "array",
                            "description": "镜头列表（必填）",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "string",
                                        "description": "画面内容描述（必填）"
                                    },
                                    "shotType": {
                                        "type": "string",
                                        "description": "景别（远景/全景/中景/近景/特写）"
                                    },
                                    "duration": {
                                        "type": "number",
                                        "description": "预估时长（秒）"
                                    },
                                    "dialogue": {
                                        "type": "string",
                                        "description": "台词/旁白，格式为'角色名：台词内容'，如'张三：你好啊'。旁白则直接写内容，无需角色名前缀"
                                    },
                                    "sound": {
                                        "type": "string",
                                        "description": "声音描述"
                                    },
                                    "soundEffect": {
                                        "type": "string",
                                        "description": "音效"
                                    },
                                    "music": {
                                        "type": "string",
                                        "description": "配乐建议"
                                    },
                                    "cameraMovement": {
                                        "type": "string",
                                        "description": "镜头运动（推/拉/摇/移等）"
                                    },
                                    "cameraAngle": {
                                        "type": "string",
                                        "description": "镜头角度（平视/俯视/仰视等）"
                                    },
                                    "cameraEquipment": {
                                        "type": "string",
                                        "description": "摄像机装备"
                                    },
                                    "focalLength": {
                                        "type": "string",
                                        "description": "镜头焦段"
                                    },
                                    "transition": {
                                        "type": "string",
                                        "description": "转场效果"
                                    },
                                    "sceneExpectation": {
                                        "type": "string",
                                        "description": "画面期望描述"
                                    },
                                    "characterIds": {
                                        "type": "array",
                                        "items": {
                                            "type": "integer"
                                        },
                                        "description": "出场角色子资产ID列表（AssetItem.id），从 list_project_assets 或 query_asset_items 获得"
                                    },
                                    "sceneAssetItemId": {
                                        "type": "integer",
                                        "description": "场景子资产ID（AssetItem.id）"
                                    },
                                    "propIds": {
                                        "type": "array",
                                        "items": {
                                            "type": "integer"
                                        },
                                        "description": "道具子资产ID列表（AssetItem.id）"
                                    },
                                    "remark": {
                                        "type": "string",
                                        "description": "备注"
                                    }
                                },
                                "required": [
                                    "content"
                                ]
                            },
                            "minItems": 1,
                            "maxItems": 500
                        },
                        "expectedParentVersion": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "调用方读取的父集合版本：创建场次用分镜集 version，插入镜头用场次 version"
                        }
                    },
                    "required": [
                        "storyboardId",
                        "storyboardEpisodeId",
                        "sceneNumber",
                        "shots",
                        "expectedParentVersion"
                    ],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return tools.execute(getToolName(), toolInput, context);
    }
}
