package com.stonewu.fusion.service.ai.tool.asset;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 查询资产详情工具（get_asset）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class AssetQueryToolExecutor implements ToolExecutor {

    private final AssetToolQueries queries;


    @Override
    public String getToolName() {
        return "get_asset";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询资产详情";
    }

    @Override
    public String getToolDescription() {
        return """
                查询指定资产的详细信息，包括名称、描述、自定义属性、封面图等。
                返回该资产的完整信息，便于了解资产的具体特征。

                如需查看资产的子资产（图片、变体等），请使用 query_asset_items 工具。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "assetId": {
                            "type": "integer",
                            "description": "资产ID（必填）"
                        }
                    },
                    "required": ["assetId"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return queries.execute(getToolName(), toolInput, context);
    }
}
