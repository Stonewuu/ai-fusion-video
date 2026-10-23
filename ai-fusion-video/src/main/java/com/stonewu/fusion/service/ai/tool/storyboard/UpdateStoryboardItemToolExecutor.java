package com.stonewu.fusion.service.ai.tool.storyboard;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 通用更新分镜镜头工具。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateStoryboardItemToolExecutor implements ToolExecutor {
    private final StoryboardToolCommands tools;


    @Override
    public String getToolName() {
        return "update_storyboard_item";
    }

    @Override
    public String getDisplayName() {
        return "更新分镜镜头";
    }

    @Override
    public String getToolDescription() {
        return "通用更新分镜镜头的画面、声音、对白、运镜、素材关联、排序和状态字段。首尾帧可继续使用专用工具。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "storyboardItemId": {
                            "type": "integer",
                            "description": "分镜镜头ID"
                        },
                        "shotNumber": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "镜头编号"
                        },
                        "imageUrl": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "镜头图片URL"
                        },
                        "referenceImageUrl": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "参考图片URL"
                        },
                        "videoUrl": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "镜头视频URL"
                        },
                        "generatedImageUrl": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "生成图片URL"
                        },
                        "generatedVideoUrl": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "生成视频URL"
                        },
                        "videoPrompt": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "视频提示词"
                        },
                        "shotType": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "景别"
                        },
                        "duration": {
                            "type": [
                                "number",
                                "null"
                            ],
                            "description": "镜头时长"
                        },
                        "content": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "画面内容"
                        },
                        "sceneExpectation": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "场景预期"
                        },
                        "sound": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "声音描述"
                        },
                        "dialogue": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "对白"
                        },
                        "soundEffect": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "音效"
                        },
                        "music": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "音乐"
                        },
                        "cameraMovement": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "运镜"
                        },
                        "cameraAngle": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "机位角度"
                        },
                        "cameraEquipment": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "摄影设备"
                        },
                        "focalLength": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "焦距"
                        },
                        "transition": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "转场"
                        },
                        "characterIds": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "minimum": 1
                            },
                            "description": "原生子资产 ID 数组；空数组清空关联"
                        },
                        "sceneAssetItemId": {
                            "type": [
                                "integer",
                                "null"
                            ],
                            "description": "场景子资产ID"
                        },
                        "propIds": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "minimum": 1
                            },
                            "description": "原生子资产 ID 数组；空数组清空关联"
                        },
                        "remark": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "备注"
                        },
                        "customData": {
                            "type": [
                                "object",
                                "null"
                            ],
                            "description": "原生扩展属性对象或明确清空"
                        },
                        "aiGenerated": {
                            "type": "boolean",
                            "description": "是否由AI生成"
                        },
                        "status": {
                            "type": "integer",
                            "description": "状态"
                        },
                        "expectedVersion": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "调用方实际读取的对象版本；冲突时不得换用新版本强制覆盖"
                        },
                        "firstFrameImageUrl": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "明确更新或清空；缺省保持原值"
                        },
                        "lastFrameImageUrl": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "明确更新或清空；缺省保持原值"
                        },
                        "firstFramePrompt": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "明确更新或清空；缺省保持原值"
                        },
                        "lastFramePrompt": {
                            "type": [
                                "string",
                                "null"
                            ],
                            "description": "明确更新或清空；缺省保持原值"
                        }
                    },
                    "required": [
                        "storyboardItemId",
                        "expectedVersion"
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
