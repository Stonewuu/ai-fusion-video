package com.stonewu.fusion.service.ai.tool.storyboard;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 查询分镜场次镜头列表工具（get_storyboard_scene_items）
 * <p>
 * 返回指定场次下的所有镜头详情，含完整的镜头信息（画面内容、运镜、对白、图片、视频等）。
 * 也支持通过 storyboardItemId 查询该镜头所在场次的所有镜头（用于获取上下文）。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class GetStoryboardSceneItemsToolExecutor implements ToolExecutor {
    private final StoryboardToolQueries tools;


    @Override
    public String getToolName() {
        return "get_storyboard_scene_items";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询场次镜头列表";
    }

    @Override
    public String getToolDescription() {
        return """
                查询分镜场次下的所有镜头详情。支持两种查询方式：
                1. 通过 sceneId 直接查询场次下的所有镜头
                2. 通过 storyboardItemId 查询该镜头所在场次的所有镜头（自动定位场次）

                返回的每个镜头包含完整信息：画面内容、景别、运镜、对白、音效、首尾帧图片URL、视频URL等。
                可用于获取上下文信息（上一个/下一个镜头），以便生成连贯的视频提示词。

                **资产引用解析**：每个镜头的 characterIds、propIds、sceneAssetItemId 会自动解析为带图片URL的资产引用信息，
                返回在 characterRefs、propRefs、sceneRef 字段中，包含子资产ID、名称、类型和图片URL，无需额外调用 query_asset_items。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "storyboardSceneId": {
                            "type": "integer",
                            "description": "分镜场次ID，直接查询该场次的所有镜头"
                        },
                        "storyboardItemId": {
                            "type": "integer",
                            "description": "分镜条目ID，自动找到所在场次并返回该场次所有镜头"
                        }
                    },
                    "anyOf": [
                        {
                            "required": [
                                "storyboardSceneId"
                            ]
                        },
                        {
                            "required": [
                                "storyboardItemId"
                            ]
                        }
                    ]
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
