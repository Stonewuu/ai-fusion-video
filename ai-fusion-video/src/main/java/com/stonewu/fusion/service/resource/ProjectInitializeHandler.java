package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.ProjectWorkspaceRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class ProjectInitializeHandler implements ResourceCommandHandler {
    private final CanvasAccessPolicy access;
    private final ProjectWorkspaceRepository projects;
    private final ResourceReadRepository resources;
    private final CanvasJson json;

    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.PROJECT;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.INITIALIZE_WORKSPACE;
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        long id = ((InitializeProjectWorkspaceCommand) input).resourceId();
        access.requireProject(id, context.actor());
        return id;
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (InitializeProjectWorkspaceCommand) input;
        var header = resources.require(CanvasValueType.PROJECT, command.resourceId(), true);
        ResourceMutations.requireVersion(header, command.expectedVersion());
        var before = resources.data(header);
        var result = projects.ensure(header, true);
        return new ResourceMutation(result.result(), result.changes(),
                json.object(Map.of("previousProjectVersion", header.version(),
                        "created", result.changes(), "resources", List.of(Map.of("header", header, "data", before)))));
    }

    @Override
    @CacheEvict(value = {"project", "script", "storyboard", "resourceDocument"}, allEntries = true)
    public void evict() {
    }
}
