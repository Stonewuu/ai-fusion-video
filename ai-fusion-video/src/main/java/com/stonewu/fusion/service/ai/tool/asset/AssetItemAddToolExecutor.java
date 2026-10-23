package com.stonewu.fusion.service.ai.tool.asset;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 添加资产图片工具（add_asset_item）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class AssetItemAddToolExecutor implements ToolExecutor {

    private final AssetToolCommands commands;

    /**
     * 来源类型：AI生成
     */
    private static final int SOURCE_AI_GENERATED = 2;


    @Override
    public String getToolName() {
        return "add_asset_item";
    }

    @Override
    public String getDisplayName() {
        return "添加子资产";
    }

    @Override
    public String getToolDescription() {
        return """
                将图片添加到现有资产中。适用于：
                - 将 AI 生成的图片保存到角色立绘
                - 为场景添加不同视角的图片
                - 为道具添加细节图

                使用场景：
                1. 调用 generate_image 生成图片后，使用此工具将图片保存到资产中
                2. 用户提供了外部图片URL，需要添加到资产中

                注意：此工具需要已存在的资产ID，请先通过 get_asset 或上下文获取目标资产ID。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "parentVersion": { "type": "integer", "minimum": 0, "description": "读取到的目标主资产版本" },
                        "assetId": {
                            "type": "integer",
                            "description": "目标资产ID（必填），将图片添加到哪个资产下"
                        },
                        "imageUrl": {
                            "type": "string",
                            "description": "图片URL（必填），通常来自 generate_image 工具的返回结果"
                        },
                        "itemType": {
                            "type": "string",
                            "description": "子资产类型（可选），可用值：front(正面)、side(侧面)、back(背面)、detail(细节)、expression(表情)、pose(姿势)、variant(变体)、original(原始)。默认为 original"
                        },
                        "name": {
                            "type": "string",
                            "description": "子资产名称（可选），如'正面立绘'、'夜景图'等"
                        },
                        "aiPrompt": {
                            "type": "string",
                            "description": "AI生成时使用的提示词（可选），便于后续追溯和重新生成"
                        }
                    },
                    "required": ["parentVersion", "assetId", "imageUrl"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return commands.execute(getToolName(), toolInput, context);
    }
}
