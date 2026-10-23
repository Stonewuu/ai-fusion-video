package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.AssetMutationRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET_ITEM;

@Component
@RequiredArgsConstructor
public class AssetBatchCreateHandler implements ResourceCommandHandler {
    private final AssetCommandSupport support;
    private final AssetMutationRepository writes;
    private final AssetResourceCreateHandler creator;
    private final AssetResourceQueryService queries;
    private final ResourceQueryService resources;
    private final AssetFields fields;
    private final CanvasJson json;

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
        return ResourceOperation.BATCH_CREATE_ASSETS;
    }

    @Override
    public void evict() {
        support.evict();
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        var command = (AssetBatchCreateCommand) input;
        if (command.resourceType() == ASSET) {
            if (command.assetId() != null || command.expectedParentVersion() != null) {
                throw AssetFields.invalid("主资产批量创建不能指定父资产");
            }
            support.authorizeCreate(command.projectId(), context);
            if (previous != null) {
                for (var asset : previous.result().path("allAssets")) {
                    support.authorize(ASSET, asset.path("id").longValue(), context, false);
                }
            }
            return command.projectId();
        }
        if (command.assetId() == null || command.expectedParentVersion() == null) {
            throw AssetFields.invalid("子资产批量创建必须指定 assetId 和 expectedParentVersion");
        }
        var parent = support.authorize(ASSET, command.assetId(), context, false);
        if (command.projectId() != null && !Objects.equals(parent.projectId(), command.projectId())) {
            throw AssetFields.invalid("资产不属于指定项目");
        }
        return parent.projectId();
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (AssetBatchCreateCommand) input;
        int parentVersion = 0;
        String parentType = null;
        if (command.resourceType() == ASSET) {
            writes.lockCreation(command.projectId(), context.actor());
        } else {
            var parent = writes.lock(ASSET, command.assetId());
            parent.requireParentVersion(command.expectedParentVersion());
            parentVersion = parent.asset.version();
            parentType = parent.data(parent.asset).path("type").textValue();
        }
        List<ObjectNode> created = new ArrayList<>();
        List<ObjectNode> existing = new ArrayList<>();
        List<ResourceChange> changes = new ArrayList<>();
        List<ObjectNode> undo = new ArrayList<>();
        for (ObjectNode entry : command.entries()) {
            ObjectNode checked = entry.deepCopy();
            if (command.resourceType() == ASSET) {
                checked.remove("initialItem");
            }
            fields.values(command.resourceType(), checked, true);
            String assetType = command.resourceType() == ASSET ? entry.path("type").textValue() : parentType;
            fields.validateProperties(assetType, checked.get("properties"));
            if (command.resourceType() == ASSET && entry.has("initialItem")) {
                if (!(entry.get("initialItem") instanceof ObjectNode initial)) {
                    throw AssetFields.invalid("initialItem 必须为对象");
                }
                fields.values(ASSET_ITEM, initial, true);
                fields.validateProperties(assetType, initial.get("properties"));
            }
            if (!entry.path("name").isTextual() || entry.path("name").textValue().isBlank()) {
                throw AssetFields.invalid("批量条目必须包含名称");
            }
            List<Long> matches = !command.reuseExisting() ? List.of() : command.resourceType() == ASSET
                    ? writes.namedAssets(command.projectId(), context.actor(), entry.path("type").textValue(),
                    entry.path("name").textValue())
                    : writes.namedItems(command.assetId(), entry.path("name").textValue());
            if (matches.size() > 1) {
                throw AssetFields.invalid("名称匹配多个资源，请用实际 ID 选择对象，或明确创建独立副本");
            }
            if (!matches.isEmpty()) {
                existing.add(command.resourceType() == ASSET ? queries.tree(matches.getFirst(), context.actor())
                        : resources.get(ASSET_ITEM, matches.getFirst(), context.actor()).data());
                continue;
            }
            ObjectNode values = entry.deepCopy();
            if (command.resourceType() == ASSET_ITEM) {
                values.put("assetId", command.assetId()).put("parentVersion", parentVersion);
            }
            ResourceMutation mutation = creator.apply(
                    new ResourceCreateCommand(command.requestId(), command.resourceType(), command.projectId(), values),
                    context);
            created.add(mutation.result());
            changes.addAll(mutation.changes());
            undo.add(mutation.previousValues());
            if (command.resourceType() == ASSET_ITEM) {
                parentVersion = mutation.result().path("assetVersion").intValue();
            }
        }
        List<ObjectNode> all = new ArrayList<>(created);
        all.addAll(existing);
        if (command.resourceType() == ASSET_ITEM) {
            int committedParentVersion = parentVersion;
            all.forEach(item -> item.put("assetVersion", committedParentVersion));
        }
        String key = command.resourceType() == ASSET ? "allAssets" : "allItems";
        ObjectNode result = json.object(Map.of("created", created, "existing", existing, key, all,
                "createdCount", created.size(), "existingCount", existing.size()));
        if (command.resourceType() == ASSET_ITEM) {
            result.put("assetId", command.assetId()).put("assetVersion", parentVersion);
        }
        return new ResourceMutation(result, changes, json.object(Map.of("operations", undo)));
    }
}
