package com.stonewu.fusion.service.storyboard.dto;

import java.util.List;

/**
 * 分镜条目资产关联局部更新命令。
 */
public record StoryboardItemAssetsPatch(
        boolean characterIdsPresent,
        List<Long> characterIds,
        boolean sceneAssetItemIdPresent,
        Long sceneAssetItemId,
        boolean propIdsPresent,
        List<Long> propIds
) {

    public boolean hasUpdates() {
        return characterIdsPresent || sceneAssetItemIdPresent || propIdsPresent;
    }
}
