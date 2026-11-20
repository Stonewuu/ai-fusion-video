package com.stonewu.fusion.repository.resource;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.asset.Asset;
import com.stonewu.fusion.entity.asset.AssetItem;
import com.stonewu.fusion.mapper.asset.AssetItemMapper;
import com.stonewu.fusion.mapper.asset.AssetMapper;
import com.stonewu.fusion.mapper.asset.AssetMapper.AccessFilter;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

@Repository
@RequiredArgsConstructor
public class AssetReadRepository {
    private final AssetMapper assets;
    private final AssetItemMapper items;
    private final ResourceReadRepository resourceRows;

    public List<ObjectNode> details(CanvasActor actor, List<Long> projects, Long projectId, String type, String keyword,
                                    Integer page, Integer size) {
        var pagination = page != null && size != null ? new Page<>(page, size, false) : null;
        return resourceRows.rows(assets.accessibleDetails(pagination,
                filter(actor, projects, projectId, type, keyword)));
    }

    public List<ObjectNode> itemDetails(List<Long> assetIds) {
        return resourceRows.rows(items.detailsByAssetIds(assetIds));
    }

    public List<ObjectNode> summaries(CanvasActor actor, List<Long> projects, Long projectId, String type,
                                      String keyword, int page, int size) {
        return resourceRows.rows(assets.accessibleSummaries(new Page<>(page, size, false),
                filter(actor, projects, projectId, type, keyword)));
    }

    public List<ObjectNode> itemSummaries(long assetId, int page, int size) {
        return resourceRows.rows(items.summaries(new Page<>(page, size, false), assetId));
    }

    public List<ObjectNode> summariesByIds(List<Long> ids, CanvasActor actor, List<Long> projects) {
        return resourceRows.rows(assets.summariesByIds(ids, filter(actor, projects, null, null, null)));
    }

    public List<ObjectNode> itemSummariesByAssetIds(List<Long> assetIds, int page, int size) {
        long offset = (long) (page - 1) * size;
        return resourceRows.rows(items.summariesByAssetIds(assetIds, offset, offset + size));
    }

    public Set<Long> existingIds(List<Long> ids) {
        return assets.selectList(new LambdaQueryWrapper<Asset>().select(Asset::getId).in(Asset::getId, ids))
                .stream().map(Asset::getId).collect(Collectors.toSet());
    }

    public ObjectNode summary(long assetId) {
        return resourceRows.row(assets.summary(assetId));
    }

    public long itemCount(long assetId) {
        return items.selectCount(new LambdaQueryWrapper<AssetItem>().eq(AssetItem::getAssetId, assetId));
    }

    public long count(CanvasActor actor, List<Long> projects, Long projectId, String type, String keyword) {
        return assets.countAccessible(filter(actor, projects, projectId, type, keyword));
    }

    public Map<String, Long> counts(CanvasActor actor, List<Long> projects, Long projectId, String keyword) {
        Map<String, Long> result = new LinkedHashMap<>();
        assets.countAccessibleTypes(filter(actor, projects, projectId, null, keyword))
                .forEach(row -> result.put(row.type(), row.total()));
        return result;
    }

    public List<Long> items(long assetId) {
        return items.selectList(new LambdaQueryWrapper<AssetItem>().select(AssetItem::getId)
                        .eq(AssetItem::getAssetId, assetId).orderByAsc(AssetItem::getSortOrder, AssetItem::getId))
                .stream().map(AssetItem::getId).toList();
    }

    private AccessFilter filter(CanvasActor actor, List<Long> projects, Long projectId, String type, String keyword) {
        String escaped = keyword == null ? null : "%" + keyword.replace("!", "!!").replace("%", "!%")
                .replace("_", "!_") + "%";
        return new AccessFilter(actor.userId(), actor.teamId(), projects, projectId, type, escaped);
    }
}
