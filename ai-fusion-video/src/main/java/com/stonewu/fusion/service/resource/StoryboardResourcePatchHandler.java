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
public class StoryboardResourcePatchHandler implements ResourceCommandHandler {
    private final StoryboardCommandSupport support;
    private final StoryboardMutationRepository writes;
    private final StoryboardFields fields;
    private final StoryboardRelations relations;
    private final StoryboardVersionView versions;

    @Override
    public CanvasValueType resourceType() {
        return STORYBOARD;
    }

    @Override
    public Set<CanvasValueType> resourceTypes() {
        return Set.of(STORYBOARD, STORYBOARD_EPISODE, STORYBOARD_SCENE, SHOT);
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
        return support.authorize(command.resourceType(), command.resourceId(), context, false);
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ResourcePatchCommand) input;
        var values = fields.values(command.resourceType(), command.patch(), false);
        var change = writes.lock(command.resourceType(), command.resourceId());
        ResourceMutations.requireVersion(change.target, command.expectedVersion());
        relations.validate(change.board, command.resourceType(), command.patch(), command.resourceId());
        change.patch(change.target, values);
        change.finish();
        return change.result(
                versions.attach(command.resourceType(), change.current(command.resourceType(), command.resourceId())));
    }
}
