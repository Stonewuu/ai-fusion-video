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

/** 通用更新子资产工具。 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateAssetItemToolExecutor implements ToolExecutor {

    private final AssetService assetService;

    @Override
    public String getToolName() {
        return "update_asset_item";
    }

    @Override
    public String getDisplayName() {
        return "更新子资产";
    }

    @Override
    public String getToolDescription() {
        return "更新子资产的名称、类型、图片、属性、排序、来源或 AI 提示词，不允许将子资产移动到其他主资产。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "assetItemId": {"type": "integer", "description": "子资产ID"},
                        "itemType": {"type": "string", "description": "子资产类型"},
                        "name": {"type": "string", "description": "子资产名称"},
                        "imageUrl": {"type": "string", "description": "图片URL"},
                        "thumbnailUrl": {"type": "string", "description": "缩略图URL"},
                        "properties": {"type": "string", "description": "扩展属性JSON"},
                        "sortOrder": {"type": "integer", "description": "排序值"},
                        "sourceType": {"type": "integer", "description": "来源类型"},
                        "aiPrompt": {"type": "string", "description": "AI生成提示词"}
                    },
                    "required": ["assetItemId"],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long itemId = params.getLong("assetItemId");
            if (itemId == null) {
                return error("缺少 assetItemId");
            }
            if (!hasUpdate(params)) {
                return error("至少提供一个需要更新的字段");
            }

            AssetItem existing = assetService.getItemById(itemId);
            Asset asset = assetService.getById(existing.getAssetId());
            if (!assetService.canAccessAsset(asset, context.getUserId())) {
                return error("无权更新该子资产");
            }

            AssetItem patch = new AssetItem();
            patch.setId(itemId);
            patch.setSortOrder(null);
            patch.setSourceType(null);
            if (params.containsKey("itemType")) patch.setItemType(params.getStr("itemType"));
            if (params.containsKey("name")) patch.setName(params.getStr("name"));
            if (params.containsKey("imageUrl")) patch.setImageUrl(params.getStr("imageUrl"));
            if (params.containsKey("thumbnailUrl")) patch.setThumbnailUrl(params.getStr("thumbnailUrl"));
            if (params.containsKey("properties")) patch.setProperties(params.getStr("properties"));
            if (params.containsKey("sortOrder")) patch.setSortOrder(params.getInt("sortOrder"));
            if (params.containsKey("sourceType")) patch.setSourceType(params.getInt("sourceType"));
            if (params.containsKey("aiPrompt")) patch.setAiPrompt(params.getStr("aiPrompt"));

            assetService.updateItem(patch);
            return JSONUtil.createObj()
                    .set("status", "success")
                    .set("assetItem", assetService.getItemById(itemId))
                    .toString();
        } catch (Exception e) {
            log.error("更新子资产失败", e);
            return error("更新子资产失败: " + e.getMessage());
        }
    }

    private boolean hasUpdate(JSONObject params) {
        return params.containsKey("itemType")
                || params.containsKey("name")
                || params.containsKey("imageUrl")
                || params.containsKey("thumbnailUrl")
                || params.containsKey("properties")
                || params.containsKey("sortOrder")
                || params.containsKey("sourceType")
                || params.containsKey("aiPrompt");
    }

    private String error(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
