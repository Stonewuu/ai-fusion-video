package com.stonewu.fusion.service.ai.tool.asset;

import cn.hutool.core.util.StrUtil;
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

import java.util.List;

/**
 * 更新子资产图片工具（update_asset_image）
 * <p>
 * 为已存在的子资产更新图片URL，适用于AI自动生图流程中
 * 子资产已预先创建（创建资产时自动生成）的场景。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateAssetImageToolExecutor implements ToolExecutor {

    /** 来源类型：AI生成 */
    private static final int SOURCE_AI_GENERATED = 2;

    private final AssetService assetService;

    @Override
    public String getToolName() {
        return "update_asset_image";
    }

    @Override
    public String getDisplayName() {
        return "更新子资产图片";
    }

    @Override
    public String getToolDescription() {
        return """
                更新已有子资产的图片。适用于：
                - 为已存在但尚无图片的子资产补充AI生成的图片
                - 替换子资产的已有图片为新生成的版本

                使用场景：
                1. AI 自动生图流程中，子资产在创建资产时已自动生成，只需更新其 imageUrl
                2. 用户要求重新生成某个子资产的图片

                注意：
                - itemId 必须属于指定的 assetId 下的子资产
                - 此工具会设置 sourceType 为 AI生成
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
                            "description": "主资产ID（必填），子资产所属的主资产"
                        },
                        "itemId": {
                            "type": "integer",
                            "description": "子资产ID（必填），要更新图片的子资产"
                        },
                        "imageUrl": {
                            "type": "string",
                            "description": "图片URL（必填），通常来自 generate_image 工具的返回结果"
                        },
                        "aiPrompt": {
                            "type": "string",
                            "description": "AI生成时使用的提示词（可选），便于后续追溯和重新生成"
                        }
                    },
                    "required": ["assetId", "itemId", "imageUrl"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long assetId = params.getLong("assetId");
            Long itemId = params.getLong("itemId");
            String imageUrl = params.getStr("imageUrl");
            String aiPrompt = params.getStr("aiPrompt");

            // 参数验证
            if (assetId == null) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "缺少必要参数: assetId").toString();
            }
            if (itemId == null) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "缺少必要参数: itemId").toString();
            }
            if (StrUtil.isBlank(imageUrl)) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "缺少必要参数: imageUrl").toString();
            }

            Long userId = context.getUserId();

            Asset asset = assetService.getById(assetId);
            if (!assetService.canAccessAsset(asset, userId)) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "未找到ID为 " + assetId + " 的资产或无权访问").toString();
            }

            // 验证子资产归属关系
            List<AssetItem> items = assetService.listItems(assetId);
            AssetItem targetItem = items.stream()
                    .filter(item -> itemId.equals(item.getId()))
                    .findFirst()
                    .orElse(null);

            if (targetItem == null) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "子资产ID " + itemId + " 不属于资产ID " + assetId).toString();
            }

            // 更新子资产图片
            targetItem.setImageUrl(imageUrl);
            targetItem.setSourceType(SOURCE_AI_GENERATED);
            if (StrUtil.isNotBlank(aiPrompt)) {
                targetItem.setAiPrompt(aiPrompt);
            }

            assetService.updateItem(targetItem);

            log.info("[update_asset_image] 子资产图片已更新 - assetId={}, itemId={}, itemType={}, userId={}",
                    assetId, itemId, targetItem.getItemType(), userId);

            return JSONUtil.createObj()
                    .set("assetId", assetId)
                    .set("assetName", asset.getName())
                    .set("itemId", itemId)
                    .set("itemType", targetItem.getItemType())
                    .set("message", String.format("子资产「%s」的图片已成功更新",
                            StrUtil.blankToDefault(targetItem.getName(), targetItem.getItemType())))
                    .toString();
        } catch (Exception e) {
            log.error("[update_asset_image] 更新子资产图片失败", e);
            return JSONUtil.createObj().set("status", "error")
                    .set("message", "更新失败: " + e.getMessage()).toString();
        }
    }
}
