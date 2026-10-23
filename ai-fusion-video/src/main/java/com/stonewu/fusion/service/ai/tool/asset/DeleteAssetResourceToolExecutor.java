package com.stonewu.fusion.service.ai.tool.asset;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 删除资产模块资源工具。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DeleteAssetResourceToolExecutor implements ToolExecutor {

    private final AssetToolCommands commands;

    private static final String RESOURCE_ASSET = "asset";
    private static final String RESOURCE_ITEM = "item";


    @Override
    public String getToolName() {
        return "delete_asset_resource";
    }

    @Override
    public String getDisplayName() {
        return "删除资产资源";
    }

    @Override
    public String getToolDescription() {
        return "删除主资产或子资产。resourceType 为 asset 时删除主资产，为 item 时删除子资产；属于高风险操作。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "expectedVersion": { "type": "integer", "minimum": 0, "description": "修改/删除对象刚刚读取的 version，图片回填使用生成前版本" },
                        "expectedParentVersion": { "type": "integer", "minimum": 0, "description": "主资产版本 assetVersion，图片回填使用生成前版本；删除主资产不传" },
                        "resourceType": {
                            "type": "string",
                            "enum": ["asset", "item"],
                            "description": "要删除的资源类型"
                        },
                        "resourceId": {"type": "integer", "description": "要删除的资源ID"}
                    },
                    "required": ["expectedVersion", "resourceType", "resourceId"],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return commands.execute(getToolName(), toolInput, context);
    }
}
