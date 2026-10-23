package com.stonewu.fusion.service.ai.tool.storyboard;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 通用更新分镜场次工具。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateStoryboardSceneToolExecutor implements ToolExecutor {
    private final StoryboardToolCommands tools;


    @Override
    public String getToolName() {
        return "update_storyboard_scene";
    }

    @Override
    public String getDisplayName() {
        return "更新分镜场次";
    }

    @Override
    public String getToolDescription() {
        return "更新分镜场次的编号、标题、地点、时间、内外景、排序或状态。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "storyboardSceneId": {
                            "type": "integer",
                            "description": "分镜场次ID"
                        },
                        "sceneNumber": {
                            "type": "string",
                            "description": "场次编号"
                        },
                        "sceneHeading": {
                            "type": "string",
                            "description": "场次标题"
                        },
                        "location": {
                            "type": "string",
                            "description": "地点"
                        },
                        "timeOfDay": {
                            "type": "string",
                            "description": "时间段"
                        },
                        "intExt": {
                            "type": "string",
                            "description": "内景或外景"
                        },
                        "status": {
                            "type": "integer",
                            "description": "状态"
                        },
                        "expectedVersion": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "调用方实际读取的对象版本；冲突时不得换用新版本强制覆盖"
                        }
                    },
                    "required": [
                        "storyboardSceneId",
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
