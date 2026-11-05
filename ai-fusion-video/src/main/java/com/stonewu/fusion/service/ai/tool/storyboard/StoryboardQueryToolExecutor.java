package com.stonewu.fusion.service.ai.tool.storyboard;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 查询分镜目录或指定一集的镜头（get_storyboard）。
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
        return "查询分镜";
    }

    @Override
    public String getToolDescription() {
        return """
                查询分镜目录，返回全部分集和场次的信息、版本与镜头数量，不返回镜头详情。
                需要镜头详情时必须提供 storyboardEpisodeId，仅返回指定一集的镜头，单次最多一集。
                只需要某一场时使用 get_storyboard_scene_items。禁止通过本工具一次查询多集或整份分镜的镜头详情。
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
                        },
                        "storyboardEpisodeId": {
                            "type": "integer",
                            "description": "可选，分镜集 ID。提供后读取该集全部镜头；省略时只返回集和场次目录。"
                        }
                    },
                    "required": [
                        "storyboardId"
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
