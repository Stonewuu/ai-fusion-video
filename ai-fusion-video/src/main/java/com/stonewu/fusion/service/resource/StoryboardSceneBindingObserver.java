package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.mapper.resource.ResourceMutationMapper;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceMutation;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

/** 在业务写入事务内检查最终状态，覆盖编辑、批量写入、恢复、导入和资产删除。 */
@Component
@RequiredArgsConstructor
public class StoryboardSceneBindingObserver implements ResourceMutationObserver {
    private final ResourceReadRepository resources;
    private final ResourceMutationMapper mutations;

    @Override
    public void validateAndApply(ResourceMutation mutation, ResourceCommandContext context) {
        Set<Long> shots = new TreeSet<>();
        for (var change : mutation.changes()) {
            if (change.resourceType() == SHOT && !change.deleted()) {
                shots.add(change.resourceId());
            }
            if (change.resourceType() == ASSET_ITEM && change.deleted()) {
                collectReferences(change.resourceId(), shots);
            }
            if (change.resourceType() == ASSET && (change.deleted() || change.changedFields().contains("type"))) {
                for (long item : mutations.lockChildren(ASSET_ITEM, "asset_id", change.resourceId())) {
                    collectReferences(item, shots);
                }
            }
        }
        for (long id : shots) {
            var shot = resources.require(SHOT, id, false);
            var data = resources.data(shot);
            if (!data.hasNonNull("sceneAssetItemId")) {
                throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "请先为镜头关联场景");
            }
            long sceneId = StoryboardFields.id(data.get("sceneAssetItemId"), "sceneAssetItemId");
            var scene = resources.find(ASSET_ITEM, sceneId, false, false).orElseThrow(this::missingScene);
            var asset = resources.find(ASSET, resources.data(scene).path("assetId").longValue(), false, false)
                    .orElseThrow(this::missingScene);
            if (!Objects.equals(asset.projectId(), shot.projectId())
                    || !"scene".equals(resources.data(asset).path("type").asText())) {
                throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "请关联当前项目的场景资产");
            }
        }
    }

    private void collectReferences(long itemId, Set<Long> shots) {
        shots.addAll(mutations.referencing(SHOT, "scene_asset_item_id", false, null, itemId, Long.toString(itemId)));
    }

    private CanvasException missingScene() {
        return new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "场景仍被镜头使用，请先更换关联场景");
    }
}
