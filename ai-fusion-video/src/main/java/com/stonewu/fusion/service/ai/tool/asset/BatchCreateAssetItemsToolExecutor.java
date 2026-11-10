package com.stonewu.fusion.service.ai.tool.asset;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;


/**
 * 批量创建子资产工具执行器
 * <p>
 * 提供 batch_create_asset_items 工具：为指定主资产批量创建子资产变体。
 * 用于分镜 pipeline 中发现剧本描述的角色变体没有对应子资产时批量创建。
 */
@Component
@RequiredArgsConstructor
public class BatchCreateAssetItemsToolExecutor implements ToolExecutor {

    private final AssetToolCommands commands;

    /**
     * 单次批量创建子资产的最大数量
     */
    private static final int MAX_BATCH_SIZE = 5;


    @Override
    public String getToolName() {
        return "batch_create_asset_items";
    }

    @Override
    public String getDisplayName() {
        return "批量创建子资产";
    }

    @Override
    public String getToolDescription() {
        return String.format("""
                为指定主资产创建外观或阶段明显不同的子资产，每次最多 %d 个。
                调用前查询已有子资产和主资产版本；reuseExisting=true 时复用唯一同名子资产，false 时创建独立副本。
                仅在服装、年龄、伤势、物体损坏或场景布局显著变化时创建。表情、小动作、对白和普通光线变化复用已有对象。
                人物变体保持主资产已经确定的身份、五官及身体特征，只描述本次明确变化；按剧情保留衣着、赤足或受伤状态，
                不为了套用描述模板添加服装、佩剑、配饰，或改写原文已确定的外貌颜色。
                子资产没有顶层 description 字段。造型说明写入 properties 中元数据对应的属性，例如人物 appearance、clothing；fieldKey 以 query_asset_metadata 为准。
                properties 只填写需要覆盖的属性；未填写的属性沿用主资产，不用“无”批量覆盖已有信息。
                本工具准备资产数据，不生成图片。返回 created、existing、allItems 以及写入后的 assetVersion，后续写入使用该版本。
                """, MAX_BATCH_SIZE);
    }

    @Override
    public String getParametersSchema() {
        return String.format("""
                {
                    "type": "object",
                    "properties": {
                        "expectedParentVersion": { "type": "integer", "minimum": 0, "description": "查询得到的主资产版本 assetVersion" },
                        "reuseExisting": { "type": "boolean", "description": "true：按名称复用唯一已有对象并返回其实际内容，创建字段不会覆盖已有内容；false：创建独立副本" },
                        "assetId": {
                            "type": "integer",
                            "description": "主资产ID（必填）"
                        },
                        "items": {
                            "type": "array",
                            "maxItems": %d,
                            "description": "子资产列表（最多%d个，超出请分次调用）。注意：仅当角色/场景的外观有显著变化时才创建，表情、情绪、简单动作变化不需要创建子资产",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "子资产名称（必填），应体现外观上的显著变化。好的示例：'手臂打石膏的张三'、'穿婚纱的李梅'、'被火烧毁的咖啡厅'。不好的示例（不应创建子资产）：'微笑的张三'、'紧张的李梅'"
                                    },
                                    "itemType": {
                                        "type": "string",
                                        "description": "子资产类型（可选），如 variant（外观变体）、costume（服装变化）、age（年龄变化）、damaged（损坏状态）等，默认为 variant"
                                    },
                                    "properties": {
                                        "type": "object",
                                        "additionalProperties": { "type": "string" },
                                        "description": "需要覆盖的子资产属性，fieldKey 来自 query_asset_metadata，例如人物 appearance、clothing；未填写的属性继承主资产"
                                    }
                                },
                                "required": ["name"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": ["expectedParentVersion", "reuseExisting", "assetId", "items"]
                }
                """, MAX_BATCH_SIZE, MAX_BATCH_SIZE);
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return commands.execute(getToolName(), toolInput, context);
    }
}
