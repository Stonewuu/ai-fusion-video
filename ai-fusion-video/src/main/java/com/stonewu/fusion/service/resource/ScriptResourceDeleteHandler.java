package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.ScriptMutationRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.SCRIPT_EPISODE;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.SCRIPT_SCENE;

@Component
@RequiredArgsConstructor
public class ScriptResourceDeleteHandler implements ResourceCommandHandler {
    private final ScriptCommandSupport support;
    private final ScriptMutationRepository writes;
    private final CanvasJson json;

    @Override
    public CanvasValueType resourceType() {
        return SCRIPT_EPISODE;
    }

    @Override
    public Set<CanvasValueType> resourceTypes() {
        return Set.of(SCRIPT_EPISODE, SCRIPT_SCENE);
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
        if (command.resourceType() == SCRIPT_EPISODE) {
            change.children(SCRIPT_SCENE, command.resourceId()).forEach(change::delete);
        }
        change.delete(change.target);
        if (command.resourceType() == SCRIPT_SCENE) {
            change.refreshEpisode();
        }
        change.refreshScript();
        return change.result(json.object(
                Map.of("id", command.resourceId(), "version", command.expectedVersion() + 1, "deleted", true)));
    }
}
