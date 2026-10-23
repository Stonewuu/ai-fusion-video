package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;

import java.util.Set;

public interface ResourceCommandHandler {
    CanvasValueType resourceType();

    default Set<CanvasValueType> resourceTypes() {
        return Set.of(resourceType());
    }

    ResourceOperation operation();

    Long authorize(ResourceCommand command, ResourceCommandContext context, ResourceCommandReceipt previous);

    ResourceMutation apply(ResourceCommand command, ResourceCommandContext context);

    void evict();
}
