package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.ScriptMutationRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class ScriptResourcePatchHandler implements ResourceCommandHandler {
    private final ScriptCommandSupport support;
    private final ScriptMutationRepository writes;
    private final ScriptFields fields;

    @Override
    public CanvasValueType resourceType() {
        return SCRIPT;
    }

    @Override
    public Set<CanvasValueType> resourceTypes() {
        return Set.of(SCRIPT, SCRIPT_EPISODE, SCRIPT_SCENE);
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
        var values = fields.values(command.resourceType(), command.patch());
        var change = writes.lock(command.resourceType(), command.resourceId());
        ResourceMutations.requireVersion(change.target, command.expectedVersion());
        change.validateAssets(command.patch());
        if (command.resourceType() == SCRIPT_EPISODE && values.containsKey("episodeNumber")) {
            change.requireEpisodeNumber((Integer) values.get("episodeNumber"), command.resourceId());
        }
        var result = change.patch(change.target, values);
        if (command.resourceType() == SCRIPT_SCENE) {
            change.refreshEpisode();
        }
        if (command.resourceType() != SCRIPT) {
            change.refreshScript();
        }
        return change.result(result);
    }
}
