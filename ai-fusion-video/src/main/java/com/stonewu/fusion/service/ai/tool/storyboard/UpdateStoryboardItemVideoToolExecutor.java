package com.stonewu.fusion.service.ai.tool.storyboard;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 更新分镜条目视频URL工具（update_storyboard_item_video）
 * <p>
 * 将生成的视频 URL 回填到分镜条目的 generatedVideoUrl 字段。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateStoryboardItemVideoToolExecutor implements ToolExecutor {
    private final StoryboardToolCommands tools;


    @Override
    public String getToolName() {
        return "update_storyboard_item_video";
    }

    @Override
    public String getDisplayName() {
        return "更新分镜视频";
    }

    @Override
    public String getToolDescription() {
        return """
                将生成的视频URL保存到分镜条目中。
                调用 generate_video 获取视频URL后，使用此工具将视频关联到对应的分镜镜头。
                """;
    }

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
                        "videoUrl": {
                            "type": "string",
                            "description": "生成的视频URL"
                        },
                        "videoPrompt": {
                            "type": "string",
                            "description": "生成视频时使用的提示词（保存以便后续复用或手动调整）"
                        },
                        "coverUrl": {
                            "type": "string",
                            "description": "视频封面图URL（可选）"
                        },
                        "expectedVersion": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "调用方实际读取的对象版本；冲突时不得换用新版本强制覆盖"
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
    public boolean isEnabled() {
        return true;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return tools.execute(getToolName(), toolInput, context);
    }
}
