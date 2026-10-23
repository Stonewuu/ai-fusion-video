package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.ProjectWorkspaceRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class ProjectCreateHandler implements ResourceCommandHandler {
    private final CanvasAccessPolicy access;
    private final ProjectWorkspaceRepository projects;

    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.PROJECT;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.CREATE;
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        ResourceCreateCommand command = (ResourceCreateCommand) input;
        if (command.projectId() != null) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "新项目不能指定所属项目");
        }
        access.requireActorScope(context.actor());
        if (previous != null) {
            access.requireProject(previous.result().path("id").longValue(), context.actor());
        }
        return null;
    }

    @Override
    public ResourceMutation apply(ResourceCommand command, ResourceCommandContext context) {
        return projects.create(((ResourceCreateCommand) command).fields(), context.actor());
    }

    @Override
    @CacheEvict(value = {"project", "script", "storyboard", "resourceDocument"}, allEntries = true)
    public void evict() {
    }
}
