package com.stonewu.fusion.service.ai.tool;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.asset.Asset;
import com.stonewu.fusion.entity.asset.AssetItem;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.asset.AssetService;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.system.SystemConfigService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * 项目资产列表工具（list_project_assets）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ListProjectAssetsToolExecutor implements ToolExecutor {

    private final AssetService assetService;
    private final ProjectService projectService;
    private final SystemConfigService systemConfigService;

    @Override
    public String getToolName() {
        return "list_project_assets";
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
                            "type": "number",
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
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long projectId = params.getLong("projectId");
            String type = params.getStr("type");
            Long userId = context.getUserId();

            List<Asset> assets;
            if (projectId != null) {
                if (!projectService.canAccessProject(projectId, userId)) {
                    return JSONUtil.createObj().set("status", "error")
                            .set("message", "无权访问该项目").toString();
                }
                assets = assetService.listByProject(projectId, type, null);
            } else {
                assets = assetService.listAccessibleByUser(userId, type);
            }

            JSONArray resultArray = new JSONArray();
            for (Asset asset : assets) {
                List<AssetItem> items = assetService.listItems(asset.getId());
                JSONArray itemsArray = new JSONArray();
                for (AssetItem item : items) {
                    itemsArray.add(JSONUtil.createObj()
                            .set("id", item.getId())
                            .set("itemType", item.getItemType())
                            .set("name", item.getName())
                            .set("imageUrl", resolvePublicUrl(item.getImageUrl()))
                            .set("thumbnailUrl", resolvePublicUrl(item.getThumbnailUrl())));
                }

                resultArray.add(JSONUtil.createObj()
                        .set("id", asset.getId())
                        .set("name", asset.getName())
                        .set("type", asset.getType())
                        .set("description", asset.getDescription())
                        .set("coverUrl", resolvePublicUrl(asset.getCoverUrl()))
                        .set("itemCount", items.size())
                        .set("items", itemsArray));
            }

            return JSONUtil.createObj()
                    .set("type", type != null ? type : "all")
                    .set("total", assets.size())
                    .set("assets", resultArray).toString();
        } catch (Exception e) {
            log.error("查询资产列表失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }

    private String resolvePublicUrl(String resourceUrl) {
        if (resourceUrl == null || resourceUrl.isBlank()) {
            return null;
        }
        String publicUrl = systemConfigService.resolvePublicUrl(resourceUrl);
        return publicUrl != null ? publicUrl : resourceUrl;
    }
}
