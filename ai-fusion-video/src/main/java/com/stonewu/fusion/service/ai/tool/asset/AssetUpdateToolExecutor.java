package com.stonewu.fusion.service.ai.tool.asset;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.asset.Asset;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.asset.AssetService;
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

    private final AssetService assetService;

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
                - 如需删除某个属性值，将其设为空字符串
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
                            "additionalProperties": { "type": "string" },
                            "description": "自定义属性键值对，与已有属性合并"
                        }
                    },
                    "required": ["assetId"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long assetId = params.getLong("assetId");
            if (assetId == null) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "缺少必要参数: assetId").toString();
            }

            Long userId = context.getUserId();

            Asset asset = assetService.getById(assetId);
            if (!assetService.canAccessAsset(asset, userId)) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "未找到ID为 " + assetId + " 的资产或无权访问").toString();
            }

            StringBuilder updatedFields = new StringBuilder();

            if (params.containsKey("name")) {
                asset.setName(params.getStr("name"));
                updatedFields.append("名称、");
            }
            if (params.containsKey("description")) {
                asset.setDescription(params.getStr("description"));
                updatedFields.append("描述、");
            }
            if (params.containsKey("properties")) {
                // 合并属性
                JSONObject newProps = params.getJSONObject("properties");
                if (asset.getProperties() != null) {
                    JSONObject existingProps = JSONUtil.parseObj(asset.getProperties());
                    existingProps.putAll(newProps);
                    asset.setProperties(existingProps.toString());
                } else {
                    asset.setProperties(newProps.toString());
                }
                updatedFields.append("属性、");
            }

            if (updatedFields.isEmpty()) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "未指定任何要更新的字段").toString();
            }

            assetService.update(asset);
            updatedFields.setLength(updatedFields.length() - 1);

            return JSONUtil.createObj()
                    .set("assetId", assetId)
                    .set("name", asset.getName())
                    .set("message", "已更新资产：" + updatedFields).toString();
        } catch (Exception e) {
            log.error("更新资产失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "更新失败: " + e.getMessage()).toString();
        }
    }
}
