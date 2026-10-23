package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.AssetMutationRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET_ITEM;

@Component
@RequiredArgsConstructor
public class AssetResourceCreateHandler implements ResourceCommandHandler {
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
        return ResourceOperation.CREATE;
    }

    @Override
    public void evict() {
        support.evict();
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        var command = (ResourceCreateCommand) input;
        if (command.resourceType() == ASSET) {
            support.authorizeCreate(command.projectId(), context);
            if (previous != null) {
                support.authorize(ASSET, previous.result().path("id").longValue(), context, false);
            }
            return command.projectId();
        }
        var parent = support.authorize(ASSET, AssetCommandSupport.id(command.fields().get("assetId"), "assetId"),
                context, false);
        if (command.projectId() != null && !Objects.equals(command.projectId(), parent.projectId())) {
            throw AssetFields.invalid("子资产与项目不匹配");
        }
        return parent.projectId();
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ResourceCreateCommand) input;
        if (command.resourceType() == ASSET) {
            return writes.createAsset(command.projectId(), command.fields(), context.actor());
        }
        long assetId = AssetCommandSupport.id(command.fields().get("assetId"), "assetId");
        int version = AssetCommandSupport.version(command.fields().get("parentVersion"), "parentVersion");
        var draft = command.fields().deepCopy();
        draft.remove(List.of("assetId", "parentVersion"));
        var values = fields.values(ASSET_ITEM, draft, true);
        var change = writes.lock(ASSET, assetId);
        change.requireParentVersion(version);
        fields.validateProperties(change.data(change.asset).path("type").textValue(), draft.get("properties"));
        values.put("assetId", assetId);
        values.putIfAbsent("itemType", "variant");
        if (!values.containsKey("sortOrder")) {
            int order = change.items().stream().map(change::data).mapToInt(item -> item.path("sortOrder").intValue())
                    .max().orElse(-1) + 1;
            if (order > 1_000_000) {
                throw AssetFields.invalid("子资产排序超过上限");
            }
            values.put("sortOrder", order);
        }
        var result = change.create(ASSET_ITEM, values);
        change.syncParent(result, values.containsKey("imageUrl"));
        result.put("assetVersion", version + 1);
        return change.result(result);
    }
}
