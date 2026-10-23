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
                列出项目下的资产列表（含子资产信息）。
                支持的资产类型：character（角色）、scene（场景）、prop（道具）

                使用场景：
                - 创建资产前先查看已有资产，避免重复创建
                - 查看角色/场景/道具的详细信息和图片

                如果提供了 projectId，则返回该项目下的资产（需有权限）。
                如果没有 projectId，则返回当前用户可访问的资产。
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
                        }
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
