package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.AssetMutationRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET_ITEM;

@Component
@RequiredArgsConstructor
public class AssetResourceDeleteHandler implements ResourceCommandHandler {
    private final AssetCommandSupport support;
    private final AssetMutationRepository writes;

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
        return ResourceOperation.DELETE;
    }

    @Override
    public void evict() {
        support.evict();
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        var command = (ResourceDeleteCommand) input;
        return support.authorize(command.resourceType(), command.resourceId(), context, previous != null).projectId();
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ResourceDeleteCommand) input;
        var change = writes.lock(command.resourceType(), command.resourceId());
        ResourceMutations.requireVersion(change.target, command.expectedVersion());
        if (command.resourceType() == ASSET_ITEM) {
            change.requireParentVersion(command.expectedParentVersion());
            return change.result(change.detachItem());
        }
        if (command.expectedParentVersion() != null) {
            throw AssetFields.invalid("主资产没有父级版本");
        }
        change.items().forEach(change::delete);
        return change.result(change.patch(change.asset, Map.of("deleted", true)));
    }
}
