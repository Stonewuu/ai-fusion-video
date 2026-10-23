package com.stonewu.fusion.repository.resource;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
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

@Repository
@RequiredArgsConstructor
public class AssetReadRepository {
    private final AssetMapper assets;
    private final AssetItemMapper items;

    public List<Long> ids(CanvasActor actor, List<Long> projects, Long projectId, String type, String keyword,
                          Integer page, Integer size) {
        var pagination = page != null && size != null ? new Page<Long>(page, size, false) : null;
        return assets.accessibleIds(pagination, filter(actor, projects, projectId, type, keyword));
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
