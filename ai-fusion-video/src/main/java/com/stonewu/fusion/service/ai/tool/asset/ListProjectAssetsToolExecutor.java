package com.stonewu.fusion.service.ai.tool.asset;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;


/**
 * 项目资产列表工具（list_project_assets）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ListProjectAssetsToolExecutor implements ToolExecutor {

    private final AssetToolQueries queries;


    @Override
    public String getToolName() {
        return "list_project_assets";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "列出资产";
    }

    @Override
    public String getToolDescription() {
        return """
                分页查询项目资产目录，仅返回 ID、名称、类型、描述摘要、版本和子项数量。
                不返回完整属性、提示词或子资产正文。默认每页 50 项，按 keyword 名称筛选可减少读取。
                支持的资产类型：character（角色）、scene（场景）、prop（道具）

                使用场景：
                - 创建资产前先查看已有资产，避免重复创建
                - 找到目标 ID 后，用 get_asset 读取主资产详情，query_asset_items 查看子资产目录

                如果提供了 projectId，则返回该项目下的资产（需有权限）。
                如果没有 projectId，则查询当前用户可访问的资产。hasMore 为 true 时用 nextPage 继续。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "projectId": {
                            "type": "integer",
                            "description": "项目ID（可选）。不传则返回当前可访问的资产"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["character", "scene", "prop"],
                            "description": "资产类型筛选（可选）"
                        },
                        "keyword": { "type": "string", "maxLength": 200, "description": "按名称筛选候选资产" },
                        "page": { "type": "integer", "minimum": 1, "description": "页码，默认 1" },
                        "pageSize": { "type": "integer", "minimum": 1, "maximum": 100, "description": "每页数量，默认 50" }
                    },
                    "required": []
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return queries.execute(getToolName(), toolInput, context);
    }
}
