package com.stonewu.fusion.service.ai.tool.storyboard;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 删除分镜模块资源工具。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DeleteStoryboardChildResourceToolExecutor implements ToolExecutor {
    private final StoryboardToolCommands tools;


    @Override
    public String getToolName() {
        return "delete_storyboard_child_resource";
    }

    @Override
    public String getDisplayName() {
        return "删除分镜子资源";
    }

    @Override
    public String getToolDescription() {
        return "删除分镜集、场次或镜头。顶层分镜由项目生命周期维护，不允许删除。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "resourceType": {
                            "type": "string",
                            "enum": [
                                "episode",
                                "scene",
                                "item"
                            ],
                            "description": "要删除的资源类型"
                        },
                        "resourceId": {
                            "type": "integer",
                            "description": "要删除的资源ID"
                        },
                        "expectedVersion": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "调用方实际读取的对象版本；冲突时不得换用新版本强制覆盖"
                        }
                    },
                    "required": [
                        "resourceType",
                        "resourceId",
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
