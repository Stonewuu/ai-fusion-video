package com.stonewu.fusion.service.canvas;

import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.resource.ResourceAccessPolicy;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class CanvasBindingPolicy {
    private final CanvasAccessPolicy canvasAccess;
    private final ResourceAccessPolicy resourceAccess;
    private final ResourceReadRepository resources;

    public void validate(CanvasDocument document, List<CanvasNode> nodes, CanvasActor actor) {
        if (document.projectId() != null) {
            canvasAccess.validateBindings(document, nodes);
            return;
        }
        for (CanvasNode node : nodes) {
            var reference = node.resource();
            if (reference == null) {
                continue;
            }
            if (reference.type() != CanvasValueType.ASSET && reference.type() != CanvasValueType.ASSET_ITEM) {
                throw invalid(node);
            }
            var resource = resources.find(reference.type(), reference.id(), false, false)
                    .orElseThrow(() -> invalid(node));
            if (resource.projectId() != null) {
                throw invalid(node);
            }
            resourceAccess.require(resource, actor);
        }
    }

    private CanvasException invalid(CanvasNode node) {
        return new CanvasException(409, CanvasErrorCode.INVALID_RESOURCE_REFERENCE,
                "个人画布仅能关联有权访问的独立素材；项目内容需在对应项目画布中使用", Map.of("nodeId", node.id()));
    }
}
