package com.stonewu.fusion.service.ai.tool.asset;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 更新子资产图片工具（update_asset_image）
 * <p>
 * 为已存在的子资产更新图片URL，适用于AI自动生图流程中
 * 子资产已预先创建（创建资产时自动生成）的场景。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateAssetImageToolExecutor implements ToolExecutor {

    private final AssetToolCommands commands;

    /**
     * 来源类型：AI生成
     */
    private static final int SOURCE_AI_GENERATED = 2;


    @Override
    public String getToolName() {
        return "update_asset_image";
    }

    @Override
    public String getDisplayName() {
        return "更新子资产图片";
    }

    @Override
    public String getToolDescription() {
        return """
                更新已有子资产的图片。适用于：
                - 为已存在但尚无图片的子资产补充AI生成的图片
                - 替换子资产的已有图片为新生成的版本

                使用场景：
                1. AI 自动生图流程中，子资产在创建资产时已自动生成，只需更新其 imageUrl
                2. 用户要求重新生成某个子资产的图片

                注意：
                - itemId 必须属于指定的 assetId 下的子资产
                - 此工具会设置 sourceType 为 AI生成
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "expectedVersion": { "type": "integer", "minimum": 0, "description": "修改/删除对象刚刚读取的 version，图片回填使用生成前版本" },
                        "expectedParentVersion": { "type": "integer", "minimum": 0, "description": "主资产版本 assetVersion，图片回填使用生成前版本；删除主资产不传" },
                        "assetId": {
                            "type": "integer",
                            "description": "主资产ID（必填），子资产所属的主资产"
                        },
                        "itemId": {
                            "type": "integer",
                            "description": "子资产ID（必填），要更新图片的子资产"
                        },
                        "imageUrl": {
                            "type": "string",
                            "description": "图片URL（必填），通常来自 generate_image 工具的返回结果"
                        },
                        "aiPrompt": {
                            "type": "string",
                            "description": "AI生成时使用的提示词（可选），便于后续追溯和重新生成"
                        }
                    },
                    "required": ["expectedVersion", "expectedParentVersion", "assetId", "itemId", "imageUrl"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return commands.execute(getToolName(), toolInput, context);
    }
}
