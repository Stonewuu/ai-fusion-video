package com.stonewu.fusion.service.ai.tool.storyboard;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 保存分镜集工具（save_storyboard_episode）
 * <p>
 * 在指定分镜脚本下创建分镜集，或按明确 ID 和版本修改已有分镜集。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SaveStoryboardEpisodeToolExecutor implements ToolExecutor {
    private final StoryboardToolCommands tools;


    @Override
    public String getToolName() {
        return "save_storyboard_episode";
    }

    @Override
    public String getDisplayName() {
        return "保存分镜集";
    }

    @Override
    public String getToolDescription() {
        return """
                在指定分镜脚本下创建分镜集，或按明确 ID 和版本修改已有分镜集。
                返回分镜集ID，后续可为该分镜集添加场次和镜头。

                【参数说明】
                - storyboardId：所属分镜脚本ID（必填）
                - scriptEpisodeId：关联的剧本分集ID（必填）
                - episodeNumber：集号，从1开始（必填）
                - title：本集标题（必填）
                - synopsis：本集梗概（可选）
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
                            "description": "所属分镜脚本ID（必填）"
                        },
                        "scriptEpisodeId": {
                            "type": "integer",
                            "description": "关联的剧本分集ID（必填）"
                        },
                        "episodeNumber": {
                            "type": "integer",
                            "description": "集号，从1开始（必填）"
                        },
                        "title": {
                            "type": "string",
                            "description": "本集标题（必填）"
                        },
                        "synopsis": {
                            "type": "string",
                            "description": "本集梗概"
                        },
                        "storyboardEpisodeId": {
                            "type": "integer",
                            "description": "明确修改已有分集时提供；缺省时只创建，不自动按剧本集匹配覆盖"
                        },
                        "expectedVersion": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "调用方实际读取的对象版本；冲突时不得换用新版本强制覆盖"
                        },
                        "expectedParentVersion": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "新建分集时必填，来自 get_storyboard 的 version"
                        }
                    },
                    "required": [
                        "storyboardId",
                        "scriptEpisodeId",
                        "episodeNumber",
                        "title"
                    ],
                    "anyOf": [
                        {
                            "required": [
                                "storyboardEpisodeId",
                                "expectedVersion"
                            ]
                        },
                        {
                            "required": [
                                "expectedParentVersion"
                            ]
                        }
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
