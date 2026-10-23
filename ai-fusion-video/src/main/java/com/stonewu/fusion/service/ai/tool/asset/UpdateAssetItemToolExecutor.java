package com.stonewu.fusion.service.ai.tool.asset;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 通用更新子资产工具。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateAssetItemToolExecutor implements ToolExecutor {

    private final AssetToolCommands commands;


    @Override
    public String getToolName() {
        return "update_asset_item";
    }

    @Override
    public String getDisplayName() {
        return "更新子资产";
    }

    @Override
    public String getToolDescription() {
        return "更新子资产的名称、类型、图片、属性、排序、来源或 AI 提示词，不允许将子资产移动到其他主资产。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "expectedVersion": { "type": "integer", "minimum": 0, "description": "修改/删除对象刚刚读取的 version，图片回填使用生成前版本" },
                        "expectedParentVersion": { "type": "integer", "minimum": 0, "description": "主资产版本 assetVersion，图片回填使用生成前版本；删除主资产不传" },
                        "assetItemId": {"type": "integer", "description": "子资产ID"},
                        "itemType": {"type": "string", "description": "子资产类型"},
                        "name": {"type": "string", "description": "子资产名称"},
                        "imageUrl": {"type": "string", "description": "图片URL"},
                        "thumbnailUrl": {"type": "string", "description": "缩略图URL"},
                        "properties": {"type": "object", "additionalProperties": true, "description": "完整扩展属性对象，替换已有属性"},
                        "sortOrder": {"type": "integer", "description": "排序值"},
                        "sourceType": {"type": "integer", "description": "来源类型"},
                        "aiPrompt": {"type": "string", "description": "AI生成提示词"}
                    },
                    "required": ["expectedVersion", "expectedParentVersion", "assetItemId"],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return commands.execute(getToolName(), toolInput, context);
    }
}
