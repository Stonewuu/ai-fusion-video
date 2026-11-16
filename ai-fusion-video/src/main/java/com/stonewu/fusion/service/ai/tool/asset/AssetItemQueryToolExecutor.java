package com.stonewu.fusion.service.ai.tool.asset;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 读取子资产详情工具（get_asset_item）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class AssetItemQueryToolExecutor implements ToolExecutor {

    public static final String SCHEMA = """
                {
                    "type": "object",
                    "properties": {
                        "assetItemId": {
                            "type": "integer",
                            "description": "子资产ID（必填）"
                        }
                    },
                    "required": ["assetItemId"]
                }
                """;

    private final AssetToolQueries queries;


    @Override
    public String getToolName() {
        return "get_asset_item";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "读取子资产详情";
    }

    @Override
    public String getToolDescription() {
        return """
                按 assetItemId 读取一个子资产的完整描述、属性、提示词和图片地址。
                query_asset_items 先返回子资产目录；确定要使用的变体后，再调用本工具。
                返回 version 与 assetVersion，分别用于子资产更新和父集合版本校验。
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
