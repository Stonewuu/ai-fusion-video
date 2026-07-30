package com.stonewu.fusion.service.ai.tool;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.asset.AssetItem;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.asset.AssetService;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * 查询分镜场次镜头列表工具（get_storyboard_scene_items）
 * <p>
 * 返回指定场次下的所有镜头详情，含完整的镜头信息（画面内容、运镜、对白、图片、视频等）。
 * 也支持通过 storyboardItemId 查询该镜头所在场次的所有镜头（用于获取上下文）。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class GetStoryboardSceneItemsToolExecutor implements ToolExecutor {

    private final StoryboardService storyboardService;
    private final AssetService assetService;

    @Override
    public String getToolName() {
        return "get_storyboard_scene_items";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询场次镜头列表";
    }

    @Override
    public String getToolDescription() {
        return """
                查询分镜场次下的所有镜头详情。支持两种查询方式：
                1. 通过 sceneId 直接查询场次下的所有镜头
                2. 通过 storyboardItemId 查询该镜头所在场次的所有镜头（自动定位场次）

                返回的每个镜头包含完整信息：画面内容、景别、运镜、对白、音效、首尾帧图片URL、视频URL等。
                可用于获取上下文信息（上一个/下一个镜头），以便生成连贯的视频提示词。

                **资产引用解析**：每个镜头的 characterIds、propIds、sceneAssetItemId 会自动解析为带图片URL的资产引用信息，
                返回在 characterRefs、propRefs、sceneRef 字段中，包含子资产ID、名称、类型和图片URL，无需额外调用 query_asset_items。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "storyboardSceneId": {
                            "type": "integer",
                            "description": "分镜场次ID，直接查询该场次的所有镜头"
                        },
                        "storyboardItemId": {
                            "type": "integer",
                            "description": "分镜条目ID，自动找到所在场次并返回该场次所有镜头"
                        }
                    }
                }
                """;
    }

    @Override
    public boolean isEnabled() {
        return true;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long storyboardSceneId = params.getLong("storyboardSceneId");
            Long storyboardItemId = params.getLong("storyboardItemId");

            if (storyboardSceneId == null && storyboardItemId == null) {
                return errorResult("请提供 storyboardSceneId 或 storyboardItemId");
            }

            // 如果提供了 storyboardItemId，先找到所在的场次
            Long targetItemId = storyboardItemId;
            if (storyboardSceneId == null) {
                StoryboardItem targetItem = storyboardService.getItemById(storyboardItemId);
                storyboardSceneId = targetItem.getStoryboardSceneId();
                if (storyboardSceneId == null) {
                    return errorResult("该分镜条目没有关联场次");
                }
            }

            // 查询场次信息
            StoryboardScene scene = storyboardService.getSceneById(storyboardSceneId);

            // 查询该场次下的所有镜头
            List<StoryboardItem> items = storyboardService.listItemsByScene(storyboardSceneId);

            // 收集所有镜头中引用的子资产ID，批量查询
            Set<Long> allAssetItemIds = new LinkedHashSet<>();
            for (StoryboardItem item : items) {
                collectAssetItemIds(allAssetItemIds, item.getCharacterIds());
                collectAssetItemIds(allAssetItemIds, item.getPropIds());
                if (item.getSceneAssetItemId() != null) {
                    allAssetItemIds.add(item.getSceneAssetItemId());
                }
            }
            // 批量查询子资产信息
            Map<Long, AssetItem> assetItemMap = batchGetAssetItems(allAssetItemIds);

            JSONArray itemList = new JSONArray();
            for (StoryboardItem item : items) {
                JSONObject itemObj = JSONUtil.createObj()
                        .set("id", item.getId())
                        .set("shotNumber", item.getShotNumber())
                        .set("autoShotNumber", item.getAutoShotNumber())
                        .set("sortOrder", item.getSortOrder())
                        .set("shotType", item.getShotType())
                        .set("content", item.getContent())
                        .set("sceneExpectation", item.getSceneExpectation())
                        .set("dialogue", item.getDialogue())
                        .set("sound", item.getSound())
                        .set("soundEffect", item.getSoundEffect())
                        .set("music", item.getMusic())
                        .set("duration", item.getDuration())
                        .set("cameraMovement", item.getCameraMovement())
                        .set("cameraAngle", item.getCameraAngle())
                        .set("cameraEquipment", item.getCameraEquipment())
                        .set("focalLength", item.getFocalLength())
                        .set("transition", item.getTransition())
                        .set("imageUrl", item.getImageUrl())
                        .set("generatedImageUrl", item.getGeneratedImageUrl())
                        .set("firstFrameImageUrl", item.getFirstFrameImageUrl())
                        .set("lastFrameImageUrl", item.getLastFrameImageUrl())
                        .set("firstFramePrompt", item.getFirstFramePrompt())
                        .set("lastFramePrompt", item.getLastFramePrompt())
                        .set("videoUrl", item.getVideoUrl())
                        .set("generatedVideoUrl", item.getGeneratedVideoUrl())
                        .set("videoPrompt", item.getVideoPrompt())
                        .set("characterIds", item.getCharacterIds())
                        .set("sceneAssetItemId", item.getSceneAssetItemId())
                        .set("propIds", item.getPropIds())
                        .set("remark", item.getRemark());

                // 内联角色参考图信息
                JSONArray characterRefs = buildAssetRefs(item.getCharacterIds(), assetItemMap);
                if (!characterRefs.isEmpty()) {
                    itemObj.set("characterRefs", characterRefs);
                }

                // 内联道具参考图信息
                JSONArray propRefs = buildAssetRefs(item.getPropIds(), assetItemMap);
                if (!propRefs.isEmpty()) {
                    itemObj.set("propRefs", propRefs);
                }

                // 内联场景参考图信息
                if (item.getSceneAssetItemId() != null) {
                    AssetItem sceneAssetItem = assetItemMap.get(item.getSceneAssetItemId());
                    if (sceneAssetItem != null) {
                        itemObj.set("sceneRef", buildSingleAssetRef(sceneAssetItem));
                    }
                }

                // 标记当前目标镜头
                if (targetItemId != null && targetItemId.equals(item.getId())) {
                    itemObj.set("isCurrentTarget", true);
                }

                itemList.add(itemObj);
            }

            return JSONUtil.createObj()
                    .set("storyboardSceneId", scene.getId())
                    .set("sceneName", scene.getSceneHeading())
                    .set("totalItems", items.size())
                    .set("items", itemList)
                    .toString();

        } catch (Exception e) {
            log.error("[get_storyboard_scene_items] 查询失败", e);
            return errorResult("查询失败: " + e.getMessage());
        }
    }

    private String errorResult(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }

    /**
     * 从 JSON 数组字符串中提取子资产 ID 到集合
     */
    private void collectAssetItemIds(Set<Long> ids, String jsonArrayStr) {
        if (StrUtil.isBlank(jsonArrayStr)) return;
        try {
            JSONArray arr = JSONUtil.parseArray(jsonArrayStr);
            for (int i = 0; i < arr.size(); i++) {
                Long id = arr.getLong(i);
                if (id != null) ids.add(id);
            }
        } catch (Exception e) {
            log.warn("[get_storyboard_scene_items] 解析子资产ID列表失败: {}", jsonArrayStr, e);
        }
    }

    /**
     * 批量查询子资产信息，返回 id → AssetItem 映射
     */
    private Map<Long, AssetItem> batchGetAssetItems(Set<Long> ids) {
        Map<Long, AssetItem> map = new HashMap<>();
        for (Long id : ids) {
            try {
                AssetItem item = assetService.getItemById(id);
                map.put(id, item);
            } catch (Exception e) {
                log.warn("[get_storyboard_scene_items] 查询子资产失败: id={}", id, e);
            }
        }
        return map;
    }

    /**
     * 根据 ID 列表 JSON 构建资产引用数组
     */
    private JSONArray buildAssetRefs(String idsJson, Map<Long, AssetItem> assetItemMap) {
        JSONArray refs = new JSONArray();
        if (StrUtil.isBlank(idsJson)) return refs;
        try {
            JSONArray arr = JSONUtil.parseArray(idsJson);
            for (int i = 0; i < arr.size(); i++) {
                Long id = arr.getLong(i);
                if (id == null) continue;
                AssetItem assetItem = assetItemMap.get(id);
                if (assetItem != null) {
                    refs.add(buildSingleAssetRef(assetItem));
                }
            }
        } catch (Exception e) {
            log.warn("[get_storyboard_scene_items] 构建资产引用失败: {}", idsJson, e);
        }
        return refs;
    }

    /**
     * 构建单个子资产的引用信息
     */
    private JSONObject buildSingleAssetRef(AssetItem assetItem) {
        return JSONUtil.createObj()
                .set("assetItemId", assetItem.getId())
                .set("assetId", assetItem.getAssetId())
                .set("name", assetItem.getName())
                .set("itemType", assetItem.getItemType())
                .set("imageUrl", assetItem.getImageUrl())
                .set("thumbnailUrl", assetItem.getThumbnailUrl());
    }
}
