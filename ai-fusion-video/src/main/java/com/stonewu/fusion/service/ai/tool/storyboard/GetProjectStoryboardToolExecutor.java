package com.stonewu.fusion.service.ai.tool.storyboard;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 查询项目唯一分镜容器工具。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class GetProjectStoryboardToolExecutor implements ToolExecutor {
    private final StoryboardToolQueries tools;


    @Override
    public String getToolName() {
        return "get_project_storyboard";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询项目分镜";
    }

    @Override
    public String getToolDescription() {
        return """
                查询指定项目唯一的分镜容器。
                返回分镜ID、项目ID、关联剧本ID、描述和总时长等基本信息。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "projectId": {
                            "type": "integer",
                            "description": "项目ID（必填）"
                        }
                    },
                    "required": [
                        "projectId"
                    ]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return tools.execute(getToolName(), toolInput, context);
    }
}
