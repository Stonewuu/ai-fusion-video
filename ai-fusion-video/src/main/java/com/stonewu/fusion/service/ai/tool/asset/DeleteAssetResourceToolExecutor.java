package com.stonewu.fusion.service.ai.tool.asset;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.asset.Asset;
import com.stonewu.fusion.entity.asset.AssetItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.asset.AssetService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/** 删除资产模块资源工具。 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DeleteAssetResourceToolExecutor implements ToolExecutor {

    private static final String RESOURCE_ASSET = "asset";
    private static final String RESOURCE_ITEM = "item";

    private final AssetService assetService;

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
                        "resourceType": {
                            "type": "string",
                            "enum": ["asset", "item"],
                            "description": "要删除的资源类型"
                        },
                        "resourceId": {"type": "integer", "description": "要删除的资源ID"}
                    },
                    "required": ["resourceType", "resourceId"],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            String resourceType = params.getStr("resourceType");
            Long resourceId = params.getLong("resourceId");
            if (resourceType == null || resourceId == null) {
                return error("缺少 resourceType 或 resourceId");
            }

            Asset asset;
            if (RESOURCE_ASSET.equals(resourceType)) {
                asset = assetService.getById(resourceId);
            } else if (RESOURCE_ITEM.equals(resourceType)) {
                AssetItem item = assetService.getItemById(resourceId);
                asset = assetService.getById(item.getAssetId());
            } else {
                return error("resourceType 仅支持 asset 或 item");
            }

            if (!assetService.canAccessAsset(asset, context.getUserId())) {
                return error("无权删除该资产资源");
            }

            if (RESOURCE_ASSET.equals(resourceType)) {
                assetService.delete(resourceId);
            } else {
                assetService.deleteItem(resourceId);
            }
            return JSONUtil.createObj()
                    .set("status", "success")
                    .set("deletedResourceType", resourceType)
                    .set("deletedResourceId", resourceId)
                    .toString();
        } catch (Exception e) {
            log.error("删除资产资源失败", e);
            return error("删除资产资源失败: " + e.getMessage());
        }
    }

    private String error(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
