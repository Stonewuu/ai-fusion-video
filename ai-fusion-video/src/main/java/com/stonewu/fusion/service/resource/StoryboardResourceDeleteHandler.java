package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.StoryboardMutationRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class StoryboardResourceDeleteHandler implements ResourceCommandHandler {
    private final StoryboardCommandSupport support;
    private final StoryboardMutationRepository writes;

    @Override
    public CanvasValueType resourceType() {
        return STORYBOARD_EPISODE;
    }

    @Override
    public Set<CanvasValueType> resourceTypes() {
        return Set.of(STORYBOARD_EPISODE, STORYBOARD_SCENE, SHOT);
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
        return support.authorize(command.resourceType(), command.resourceId(), context, previous != null);
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ResourceDeleteCommand) input;
        var change = writes.lock(command.resourceType(), command.resourceId());
        ResourceMutations.requireVersion(change.target, command.expectedVersion());
        var result = change.deleteTree(change.target);
        change.finish();
        return change.result(result);
    }
}
