package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.AssetMutationRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET_ITEM;

@Component
@RequiredArgsConstructor
public class AssetResourcePatchHandler implements ResourceCommandHandler {
    private final AssetCommandSupport support;
    private final AssetMutationRepository writes;
    private final AssetFields fields;

    @Override
    public CanvasValueType resourceType() {
        return ASSET;
    }

    @Override
    public Set<CanvasValueType> resourceTypes() {
        return Set.of(ASSET, ASSET_ITEM);
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.PATCH;
    }

    @Override
    public void evict() {
        support.evict();
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        var command = (ResourcePatchCommand) input;
        return support.authorize(command.resourceType(), command.resourceId(), context, false).projectId();
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ResourcePatchCommand) input;
        var change = writes.lock(command.resourceType(), command.resourceId());
        ResourceMutations.requireVersion(change.target, command.expectedVersion());
        ObjectNode patch = command.patch().deepCopy();
        if (patch.has("propertyChanges")) {
            if (patch.has("properties") || !(patch.remove("propertyChanges") instanceof ObjectNode edits)) {
                throw AssetFields.invalid("propertyChanges 必须为对象，不能同时指定 properties");
            }
            var before = change.data(change.target).get("properties");
            ObjectNode merged;
            if (before == null || before.isNull()) {
                merged = patch.objectNode();
            } else if (before instanceof ObjectNode object) {
                merged = object.deepCopy();
            } else {
                throw AssetFields.invalid("已有属性不是对象，请先明确替换 properties");
            }
            edits.fields().forEachRemaining(entry -> {
                if (entry.getValue().isNull()) {
                    merged.remove(entry.getKey());
                } else {
                    merged.set(entry.getKey(), entry.getValue());
                }
            });
            patch.set("properties", merged);
        }
        var values = fields.values(command.resourceType(), patch, false);
        if (command.resourceType() == ASSET_ITEM) {
            change.requireParentVersion(command.expectedParentVersion());
        } else {
            if (command.expectedParentVersion() != null) {
                throw AssetFields.invalid("主资产没有父级版本");
            }
            var existing = change.data(change.asset);
            String type = patch.has("type") ? patch.path("type").textValue() : existing.path("type").textValue();
            if (patch.has("properties") || patch.has("type")) {
                fields.validateProperties(type,
                        patch.has("properties") ? patch.get("properties") : existing.get("properties"));
            }
        }
        var result = change.patch(change.target, values);
        if (command.resourceType() == ASSET_ITEM) {
            change.syncParent(result, values.containsKey("imageUrl") || values.containsKey("itemType"));
            result.put("assetVersion", change.asset.version() + 1);
        }
        return change.result(result);
    }
}
