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
                保存分镜镜头的首帧或尾帧图片及最终生图提示词，写入需要 expectedVersion。
                生图成功后传入 imageUrl 和实际使用的 framePrompt。
                仅优化提示词时只传 framePrompt，省略 imageUrl，保留已有图片。
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
                            "description": "生成成功的图片URL；仅修订提示词时省略，不传 null 或空字符串"
                        },
                        "framePrompt": {
                            "type": "string",
                            "description": "实际用于生成的最终提示词，或仅修订模式下供下次生成使用的提示词"
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
                        "expectedVersion"
                    ],
                    "anyOf": [
                        { "required": ["imageUrl"] },
                        { "required": ["framePrompt"] }
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
