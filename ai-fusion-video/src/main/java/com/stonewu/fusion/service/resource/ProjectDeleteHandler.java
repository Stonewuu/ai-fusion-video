package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.ProjectWorkspaceRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class ProjectDeleteHandler implements ResourceCommandHandler {
    private final CanvasAccessPolicy access;
    private final ProjectWorkspaceRepository projects;
    private final ResourceReadRepository resources;

    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.PROJECT;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.DELETE;
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        long id = ((ResourceDeleteCommand) input).resourceId();
        if (previous == null) {
            access.requireProjectManager(id, context.actor());
        } else {
            // 仅重放当前用户已提交删除的最小回执，不读取已删除项目的正文或恢复数据。
            access.requireActorScope(context.actor());
        }
        return id;
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ResourceDeleteCommand) input;
        var header = resources.require(CanvasValueType.PROJECT, command.resourceId(), true);
        ResourceMutations.requireVersion(header, command.expectedVersion());
        return projects.delete(header);
    }

    @Override
    @CacheEvict(value = {"project", "projectMember", "asset", "assetItem", "script", "episode", "scene", "storyboard",
            "storyboardEpisode", "storyboardScene", "storyboardItem", "storyboardStatistics", "resourceDocument"}, allEntries = true)
    public void evict() {
    }
}
