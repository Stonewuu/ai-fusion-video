package com.stonewu.fusion.service.ai.tool.asset;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 更新资产工具（update_asset）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class AssetUpdateToolExecutor implements ToolExecutor {

    private final AssetToolCommands commands;


    @Override
    public String getToolName() {
        return "update_asset";
    }

    @Override
    public String getDisplayName() {
        return "更新资产";
    }

    @Override
    public String getToolDescription() {
        return """
                更新已有资产的信息。仅传入需要修改的字段即可，未传的字段保持不变。
                可更新的字段包括：名称、描述、自定义属性（properties）等。

                **注意**：
                - 更新 properties 时，传入的 properties 会与现有 properties 进行合并（非替换）
                - 如需删除某个属性值，将该属性值设为 null
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "expectedVersion": { "type": "integer", "minimum": 0, "description": "修改/删除对象刚刚读取的 version，图片回填使用生成前版本" },
                        "assetId": {
                            "type": "integer",
                            "description": "资产ID（必填）"
                        },
                        "name": {
                            "type": "string",
                            "description": "资产名称"
                        },
                        "description": {
                            "type": "string",
                            "description": "资产描述"
                        },
                        "properties": {
                            "type": "object",
                            "additionalProperties": { "type": ["string", "null"] },
                            "description": "自定义属性键值对，与已有属性合并"
                        }
                    },
                    "required": ["expectedVersion", "assetId"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return commands.execute(getToolName(), toolInput, context);
    }
}
