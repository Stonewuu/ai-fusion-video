package com.stonewu.fusion.service.ai.tool.asset;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.asset.Asset;
import com.stonewu.fusion.entity.asset.AssetItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.asset.AssetService;
import com.stonewu.fusion.service.project.ProjectService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * 查询子资产列表工具执行器
 * <p>
 * 提供 query_asset_items 工具：查询主资产下的所有子资产。
 * 支持单个查询（assetId）和批量查询（assetIds，最多10个）。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class QueryAssetItemsToolExecutor implements ToolExecutor {

    private final AssetService assetService;
    private final ProjectService projectService;

    private static final int MAX_BATCH_SIZE = 10;

    @Override
    public String getToolName() {
        return "query_asset_items";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询子资产";
    }

    @Override
    public String getToolDescription() {
        return """
                查询主资产下的所有子资产（变体）列表。支持批量查询，一次最多查10个资产。

                使用场景：
                - 剧本转分镜时，先查询角色/场景的子资产，判断是否有可复用的变体
                - 批量生图时，一次查询多个资产的子资产列表

                **复用判断原则**：
                - 每个主资产创建时会自动生成一个初始子资产（默认变体），图片挂在子资产上
                - 子资产代表外观上有显著变化的变体（如受伤、换装、年龄变化、场景损毁等）
                - 如果剧本描述的只是表情变化（微笑、愤怒）、心理状态（紧张、兴奋）、
                  简单动作（奔跑、坐下），则直接复用初始子资产即可，无需查找或创建新子资产
                - 只有当角色/场景的外观发生了显著变化（需要一张新参考图来表达）时，
                  才需要查找是否有匹配的子资产

                三种查询方式：
                1. **批量查询**（推荐）：传 assetIds 数组，一次查询多个资产（最多10个）
                2. 按 assetId 精确查询单个
                3. 按 assetName + projectId 模糊匹配主资产名称

                返回每个资产的子资产列表，包含 id、name、description、itemType、imageUrl、thumbnailUrl。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "assetIds": {
                            "type": "array",
                            "items": { "type": "integer" },
                            "description": "主资产ID数组（批量查询，最多10个，推荐使用）",
                            "maxItems": 10
                        },
                        "assetId": {
                            "type": "integer",
                            "description": "主资产ID（单个查询）"
                        },
                        "assetName": {
                            "type": "string",
                            "description": "主资产名称（按名称模糊匹配时使用）"
                        },
                        "projectId": {
                            "type": "integer",
                            "description": "项目ID（按名称查询时用于限定范围，可选）"
                        }
                    }
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            JSONArray assetIdsArray = params.getJSONArray("assetIds");
            Long assetId = params.getLong("assetId");
            String assetName = params.getStr("assetName");
            Long projectId = params.getLong("projectId");
            Long userId = context.getUserId();

            // 批量查询模式
            if (assetIdsArray != null && !assetIdsArray.isEmpty()) {
                return executeBatch(assetIdsArray, userId);
            }

            // 单个查询模式
            return executeSingle(assetId, assetName, projectId, userId);
        } catch (Exception e) {
            log.error("查询子资产列表失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }

    /**
     * 批量查询多个资产的子资产
     */
    private String executeBatch(JSONArray assetIdsArray, Long userId) {
        List<Long> ids = new ArrayList<>();
        for (int i = 0; i < Math.min(assetIdsArray.size(), MAX_BATCH_SIZE); i++) {
            Long id = assetIdsArray.getLong(i);
            if (id != null) {
                ids.add(id);
            }
        }

        if (ids.isEmpty()) {
            return JSONUtil.createObj().set("status", "error").set("message", "assetIds 为空").toString();
        }

        log.info("[query_asset_items] 批量查询 {} 个资产: {}", ids.size(), ids);

        JSONArray results = new JSONArray();
        for (Long id : ids) {
            try {
                Asset asset = assetService.getById(id);
                if (asset == null) {
                    results.add(JSONUtil.createObj()
                            .set("assetId", id)
                            .set("status", "error")
                            .set("message", "资产不存在"));
                    continue;
                }
                if (!assetService.canAccessAsset(asset, userId)) {
                    results.add(JSONUtil.createObj()
                            .set("assetId", id)
                            .set("status", "error")
                            .set("message", "无权访问"));
                    continue;
                }

                List<AssetItem> items = assetService.listItems(asset.getId());
                results.add(buildAssetResult(asset, items));
            } catch (Exception e) {
                results.add(JSONUtil.createObj()
                        .set("assetId", id)
                        .set("status", "error")
                        .set("message", "查询失败: " + e.getMessage()));
            }
        }

        return JSONUtil.createObj()
                .set("totalAssets", results.size())
                .set("assets", results)
                .toString();
    }

    /**
     * 单个资产查询（兼容原有参数）
     */
    private String executeSingle(Long assetId, String assetName, Long projectId, Long userId) {
        Asset asset = null;

        // 方式1：按 assetId 精确查询
        if (assetId != null) {
            asset = assetService.getById(assetId);
            if (asset != null && !assetService.canAccessAsset(asset, userId)) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "未找到ID为 " + assetId + " 的资产或无权访问").toString();
            }
        }
        // 方式2：按名称模糊匹配
        else if (StrUtil.isNotBlank(assetName) && projectId != null) {
            if (!projectService.canAccessProject(projectId, userId)) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "无权访问该项目").toString();
            }
            List<Asset> assets = assetService.listByProject(projectId);
            asset = assets.stream()
                    .filter(a -> assetName.equals(a.getName()))
                    .findFirst().orElse(null);
            if (asset == null) {
                asset = assets.stream()
                        .filter(a -> a.getName() != null && a.getName().contains(assetName))
                        .findFirst().orElse(null);
            }
        }

        if (asset == null) {
            return JSONUtil.createObj().set("status", "error")
                    .set("message", "未找到匹配的资产。请提供 assetId 或 assetName + projectId").toString();
        }

        List<AssetItem> items = assetService.listItems(asset.getId());
        return buildAssetResult(asset, items).toString();
    }

    /**
     * 构建单个资产的查询结果
     */
    private JSONObject buildAssetResult(Asset asset, List<AssetItem> items) {
        JSONArray itemsArray = new JSONArray();
        for (AssetItem item : items) {
            itemsArray.add(JSONUtil.createObj()
                    .set("id", item.getId())
                    .set("name", item.getName())
                    .set("itemType", item.getItemType())
                    .set("imageUrl", item.getImageUrl())
                    .set("thumbnailUrl", item.getThumbnailUrl())
                    .set("properties", item.getProperties()));
        }

        return JSONUtil.createObj()
                .set("assetId", asset.getId())
                .set("assetName", asset.getName())
                .set("assetType", asset.getType())
                .set("assetDescription", asset.getDescription())
                .set("totalItems", items.size())
                .set("items", itemsArray);
    }
}
