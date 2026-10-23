package com.stonewu.fusion.service.ai.tool.storyboard;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class BatchUpdateStoryboardToolExecutor implements ToolExecutor {
    private final StoryboardToolCommands tools;

    @Override
    public String getToolName() {
        return "batch_update_storyboard";
    }

    @Override
    public String getDisplayName() {
        return "批量修改分镜";
    }

    @Override
    public String getToolDescription() {
        return "一次原子修改同一分镜中的多个对象。每个对象必须提供实际版本，任一版本冲突则全部不写入；排序使用集合工具。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "storyboardId": {
                            "type": "integer",
                            "minimum": 1
                        },
                        "entries": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 500,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "resourceType": {
                                        "type": "string",
                                        "enum": [
                                            "storyboard",
                                            "storyboardEpisode",
                                            "storyboardScene",
                                            "shot"
                                        ]
                                    },
                                    "resourceId": {
                                        "type": "integer",
                                        "minimum": 1
                                    },
                                    "expectedVersion": {
                                        "type": "integer",
                                        "minimum": 0
                                    },
                                    "patch": {
                                        "type": "object",
                                        "description": "仅提供需要修改的领域字段；缺省保持不变，允许清空的字段可用 null。"
                                    }
                                },
                                "required": [
                                    "resourceType",
                                    "resourceId",
                                    "expectedVersion",
                                    "patch"
                                ],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": [
                        "storyboardId",
                        "entries"
                    ],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String input, ToolExecutionContext context) {
        return tools.execute(getToolName(), input, context);
    }
}
