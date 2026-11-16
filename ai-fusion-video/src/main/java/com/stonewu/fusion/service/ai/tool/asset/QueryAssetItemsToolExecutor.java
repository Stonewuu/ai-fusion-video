package com.stonewu.fusion.service.ai.tool.asset;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 查询子资产列表工具执行器
 * <p>
 * 提供 query_asset_items 工具：查询主资产下的所有子资产。
 * 支持单个查询（assetId）和批量查询（assetIds，最多10个）。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class QueryAssetItemsToolExecutor implements ToolExecutor {

    public static final String SCHEMA = """
                {
                    "type": "object",
                    "properties": {
                        "assetIds": {
                            "type": "array",
                            "items": { "type": "integer" },
                            "description": "主资产ID数组（批量查询，最多10个，推荐使用）",
                            "maxItems": 10
                        },
                        "assetId": {
                            "type": "integer",
                            "description": "主资产ID（单个查询）"
                        },
                        "assetName": {
                            "type": "string",
                            "description": "主资产名称（按名称模糊匹配时使用）"
                        },
                        "projectId": {
                            "type": "integer",
                            "description": "按 assetName 查询时必须提供的项目ID"
                        },
                        "page": { "type": "integer", "minimum": 1, "description": "页码，默认 1" },
                        "pageSize": { "type": "integer", "minimum": 1, "maximum": 100, "description": "每个资产的子项数量，默认 20" }
                    }
                }
                """;

    private final AssetToolQueries queries;


    @Override
    public String getToolName() {
        return "query_asset_items";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询子资产";
    }

    @Override
    public String getToolDescription() {
        return """
                分页查询子资产（变体）目录，返回 ID、名称、类型、描述摘要、hasImage、version 与 assetVersion。
                默认每个资产返回 20 个子项，不返回完整属性、提示词和图片地址。
                确定目标后使用 get_asset_item 按 assetItemId 读取单个子项的完整详情。
                支持 assetIds 批量查询，每个资产的 hasMore、nextPage 独立表示其子项翻页状态。

                使用场景：
                - 剧本转分镜时，先查询角色/场景的子资产，判断是否有可复用的变体
                - 批量生图时，一次查询多个资产的子资产列表

                **复用判断原则**：
                - 每个主资产创建时会自动生成一个初始子资产（默认变体），图片挂在子资产上
                - 子资产代表外观上有显著变化的变体（如受伤、换装、年龄变化、场景损毁等）
                - 如果剧本描述的只是表情变化（微笑、愤怒）、心理状态（紧张、兴奋）、
                  简单动作（奔跑、坐下），则直接复用初始子资产即可，无需查找或创建新子资产
                - 只有当角色/场景的外观发生了显著变化（需要一张新参考图来表达）时，
                  才需要查找是否有匹配的子资产

                三种查询方式：
                1. **批量查询**（推荐）：传 assetIds 数组，一次查询多个资产（最多10个）
                2. 按 assetId 精确查询单个
                3. 按 assetName + projectId 查询主资产候选摘要；确定 assetId 后再查询子资产目录

                page 和 pageSize 用于当前查询的分页。需要核对完整主资产身份时使用 get_asset。
                """;
    }

    @Override
    public String getParametersSchema() {
        return SCHEMA;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return queries.execute(getToolName(), toolInput, context);
    }
}
