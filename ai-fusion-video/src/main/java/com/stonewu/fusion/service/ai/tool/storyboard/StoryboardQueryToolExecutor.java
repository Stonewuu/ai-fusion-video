package com.stonewu.fusion.service.ai.tool.storyboard;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 查询分镜详情工具（get_storyboard）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class StoryboardQueryToolExecutor implements ToolExecutor {
    private final StoryboardToolQueries tools;


    @Override
    public String getToolName() {
        return "get_storyboard";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询分镜详情";
    }

    @Override
    public String getToolDescription() {
        return """
                查询分镜脚本的详情，包含所有分镜条目信息。
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
                            "description": "分镜ID"
                        }
                    },
                    "required": [
                        "storyboardId"
                    ]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return tools.execute(getToolName(), toolInput, context);
    }
}
