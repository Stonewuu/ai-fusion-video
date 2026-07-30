package com.stonewu.fusion.service.ai.tool;

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
 * 查询资产详情工具（get_asset）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class AssetQueryToolExecutor implements ToolExecutor {

    private final AssetService assetService;

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
                            "type": "number",
                            "description": "资产ID（必填）"
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

            return JSONUtil.createObj()
                    .set("assetId", asset.getId())
                    .set("projectId", asset.getProjectId())
                    .set("type", asset.getType())
                    .set("name", asset.getName())
                    .set("description", asset.getDescription())
                    .set("coverUrl", asset.getCoverUrl())
                    .set("properties", asset.getProperties() != null ? JSONUtil.parseObj(asset.getProperties()) : null)
                    .set("tags", asset.getTags())
                    .set("sourceType", asset.getSourceType())
                    .set("status", asset.getStatus())
                    .toString();
        } catch (Exception e) {
            log.error("查询资产失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }
}
