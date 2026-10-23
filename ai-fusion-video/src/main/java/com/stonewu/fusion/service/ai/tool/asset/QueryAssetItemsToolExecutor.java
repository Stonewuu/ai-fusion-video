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

    private final AssetToolQueries queries;


    private static final int MAX_BATCH_SIZE = 10;

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
                查询主资产下的所有子资产（变体）列表。支持批量查询，一次最多查10个资产。

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
                3. 按 assetName + projectId 查询名称候选列表；后续写入必须使用明确 ID

                返回 assets 数组、主资产 version、子资产 version 和 assetVersion。公网地址未配置时 URL 为空，Stored 字段保留库内路径供受控媒体读取。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
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
                            "description": "项目ID（按名称查询时用于限定范围，可选）"
                        }
                    }
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return queries.execute(getToolName(), toolInput, context);
    }
}
