package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import com.stonewu.fusion.mapper.resource.ResourceMutationMapper;

import java.util.LinkedHashMap;

import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@Component
@RequiredArgsConstructor
public class ProjectResourcePatchHandler implements ResourceCommandHandler {
    private final ResourceMutationMapper mutations;
    private final CanvasAccessPolicy access;
    private final CanvasJson json;
    private final ResourceReadRepository resources;
    private final ProjectFields projectFields;

    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.PROJECT;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.PATCH;
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        ResourcePatchCommand command = (ResourcePatchCommand) input;
        access.requireProject(command.resourceId(), context.actor());
        return command.resourceId();
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        ResourcePatchCommand command = (ResourcePatchCommand) input;
        var header = resources.require(CanvasValueType.PROJECT, command.resourceId(), true);
        ResourceMutations.requireVersion(header, command.expectedVersion());
        ObjectNode before = resources.data(header);
        Map<String, Object> fields = projectFields.values(command.patch(), false);
        ObjectNode undo = json.object(Map.of("project", before));
        undo.putArray("resources").add(json.object(Map.of("header", header, "data", before)));
        Map<String, Object> columns = new LinkedHashMap<>();
        fields.forEach((name, value) -> columns.put(projectFields.column(name), value));
        int rows = mutations.patch(CanvasValueType.PROJECT, command.resourceId(), header.version(), columns, false);
        if (rows != 1) {
            throw conflict();
        }
        List<ResourceChange> changes = new ArrayList<>();
        changes.add(ResourceMutations.change(CanvasValueType.PROJECT, command.resourceId(), command.resourceId(),
                header.version() + 1, false, List.copyOf(fields.keySet())));
        if (fields.containsKey("name") && !Objects.equals(fields.get("name"), before.path("name").textValue())) {
            synchronizeTitles(CanvasValueType.SCRIPT, command.resourceId(), (String) fields.get("name"),
                    changes, undo);
            synchronizeTitles(CanvasValueType.STORYBOARD, command.resourceId(),
                    (String) fields.get("name"), changes, undo);
        }
        return new ResourceMutation(
                resources.data(resources.require(CanvasValueType.PROJECT, command.resourceId(), false)), changes, undo);
    }

    @Override
    @CacheEvict(value = {"project", "script", "storyboard", "resourceDocument"}, allEntries = true)
    public void evict() {
    }

    private void synchronizeTitles(CanvasValueType type, long projectId, String title,
                                   List<ResourceChange> changes, ObjectNode undo) {
        List<Map<String, Object>> children = mutations.lockProjectTitles(type, projectId);
        undo.set(type.code(), json.object(Map.of("items", children)).get("items"));
        for (Map<String, Object> child : children) {
            if (Objects.equals(child.get("title"), title)) {
                continue;
            }
            long id = ((Number) child.get("id")).longValue();
            int version = ((Number) child.get("version")).intValue();
            if (version == Integer.MAX_VALUE) {
                throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "业务对象版本达到上限");
            }
            var header = resources.require(type, id, false);
            undo.withArray("resources").add(json.object(Map.of("header", header, "data", resources.data(header))));
            if (mutations.patch(type, id, version, Map.of("title", title), false) != 1) {
                throw conflict();
            }
            changes.add(ResourceMutations.change(type, id, projectId, version + 1, false, List.of("title")));
        }
    }

    private CanvasException conflict() {
        return new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT, "业务内容版本已变化", Map.of());
    }
}
