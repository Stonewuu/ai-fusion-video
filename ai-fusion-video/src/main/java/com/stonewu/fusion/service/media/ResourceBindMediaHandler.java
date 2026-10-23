package com.stonewu.fusion.service.media;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.resource.ResourceCommandHandler;
import com.stonewu.fusion.service.resource.ResourceMutations;
import com.stonewu.fusion.service.resource.model.*;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
public class ResourceBindMediaHandler implements ResourceCommandHandler {
    private final ResourceReadRepository resources;
    private final ResourceCommandHandler assets;
    private final ResourceCommandHandler projects;
    private final ResourceCommandHandler storyboards;

    public ResourceBindMediaHandler(ResourceReadRepository resources,
                                    @Qualifier("assetResourcePatchHandler") ResourceCommandHandler assets,
                                    @Qualifier("projectResourcePatchHandler") ResourceCommandHandler projects,
                                    @Qualifier("storyboardResourcePatchHandler") ResourceCommandHandler storyboards) {
        this.resources = resources;
        this.assets = assets;
        this.projects = projects;
        this.storyboards = storyboards;
    }

    @Override
    public CanvasValueType resourceType() {
        return PROJECT;
    }

    @Override
    public Set<CanvasValueType> resourceTypes() {
        return Set.of(PROJECT, ASSET, ASSET_ITEM, SHOT);
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.BIND_MEDIA;
    }

    @Override
    public void evict() {
        assets.evict();
        projects.evict();
        storyboards.evict();
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        var command = (ResourceBindMediaCommand) input;
        if (command.resourceType() != ASSET_ITEM && command.expectedParentVersion() != null) {
            throw MediaException.invalid("此对象不接受父级资产版本");
        }
        return delegate(command.resourceType()).authorize(patch(command, null), context, previous);
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ResourceBindMediaCommand) input;
        var header = resources.require(command.resourceType(), command.resourceId(), false);
        ResourceMutations.requireVersion(header, command.expectedVersion());
        var before = resources.data(header);
        ObjectNode patch = before.objectNode();
        for (var binding : command.bindings().entrySet()) {
            String field = binding.getKey();
            if (!MediaSourceFields.fields(command.resourceType()).containsKey(field)) {
                throw MediaException.invalid("没有登记此媒体绑定字段：" + field);
            }
            String reference = MediaReference.of(binding.getValue());
            int separator = field.indexOf('.');
            if (separator < 0) {
                patch.put(field, reference);
            } else {
                String root = field.substring(0, separator);
                ObjectNode nested;
                if (patch.get(root) instanceof ObjectNode object) {
                    nested = object;
                } else {
                    var old = before.get(root);
                    if (old != null && !old.isNull() && !old.isObject()) {
                        throw MediaException.invalid("已有扩展字段不是对象");
                    }
                    nested = old instanceof ObjectNode object ? object.deepCopy() : before.objectNode();
                    patch.set(root, nested);
                }
                nested.put(field.substring(separator + 1), reference);
            }
        }
        // 通过既有处理器维护父版本、封面和分镜汇总，随后统一观察器验证权限、类型并登记引用。
        return delegate(command.resourceType()).apply(patch(command, patch), context);
    }

    private ResourcePatchCommand patch(ResourceBindMediaCommand command, ObjectNode patch) {
        return new ResourcePatchCommand(command.requestId(), command.resourceType(), command.resourceId(),
                command.expectedVersion(), patch, command.expectedParentVersion());
    }

    private ResourceCommandHandler delegate(CanvasValueType type) {
        return switch (type) {
            case PROJECT -> projects;
            case ASSET, ASSET_ITEM -> assets;
            case SHOT -> storyboards;
            default -> throw MediaException.invalid("此对象不支持手动绑定媒体");
        };
    }
}
