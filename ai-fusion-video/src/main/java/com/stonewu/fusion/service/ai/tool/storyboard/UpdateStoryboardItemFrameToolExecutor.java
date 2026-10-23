package com.stonewu.fusion.service.ai.tool.storyboard;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 更新分镜条目首尾帧工具（update_storyboard_item_frame）。
 * <p>
 * 将 AI 生成的图片 URL 回填到分镜条目的首帧或尾帧字段。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateStoryboardItemFrameToolExecutor implements ToolExecutor {
    private final StoryboardToolCommands tools;


    /**
     * 获取工具名称。
     *
     * @return 工具名称
     */
    @Override
    public String getToolName() {
        return "update_storyboard_item_frame";
    }

    /**
     * 获取工具展示名称。
     *
     * @return 工具展示名称
     */
    @Override
    public String getDisplayName() {
        return "更新分镜首尾帧";
    }

    /**
     * 获取工具说明。
     *
     * @return 工具说明
     */
    @Override
    public String getToolDescription() {
        return """
                将生成的图片URL保存到分镜条目的首帧或尾帧字段中。
                调用 generate_image 获取图片URL后，使用此工具将图片关联到对应的分镜镜头。
                """;
    }

    /**
     * 获取工具参数 JSON Schema。
     *
     * @return 参数 JSON Schema
     */
    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "storyboardItemId": {
                            "type": "integer",
                            "description": "分镜条目ID"
                        },
                        "frameType": {
                            "type": "string",
                            "enum": [
                                "first",
                                "last"
                            ],
                            "description": "帧类型：first-首帧，last-尾帧"
                        },
                        "imageUrl": {
                            "type": "string",
                            "description": "generate_image 返回的图片URL"
                        },
                        "framePrompt": {
                            "type": "string",
                            "description": "生成首尾帧时使用的提示词"
                        },
                        "expectedVersion": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "调用方实际读取的对象版本；冲突时不得换用新版本强制覆盖"
                        }
                    },
                    "required": [
                        "storyboardItemId",
                        "frameType",
                        "imageUrl",
                        "expectedVersion"
                    ],
                    "additionalProperties": false
                }
                """;
    }

    /**
     * 判断工具是否启用。
     *
     * @return 是否启用
     */
    @Override
    public boolean isEnabled() {
        return true;
    }

    /**
     * 执行分镜首尾帧回填。
     *
     * @param toolInput 工具入参 JSON
     * @param context   工具执行上下文
     * @return 执行结果 JSON
     */
    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return tools.execute(getToolName(), toolInput, context);
    }
}
