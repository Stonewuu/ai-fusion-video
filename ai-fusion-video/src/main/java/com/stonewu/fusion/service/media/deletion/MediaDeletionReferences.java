package com.stonewu.fusion.service.media.deletion;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.media.MediaBindingEntity;
import com.stonewu.fusion.mapper.media.MediaBindingMapper;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasCommandService;
import com.stonewu.fusion.service.canvas.command.CanvasEditCommand;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.deletion.MediaDeletionData.Reference;
import com.stonewu.fusion.service.resource.ResourceAccessPolicy;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.HashSet;
import java.util.UUID;

@Component
@RequiredArgsConstructor
public class MediaDeletionReferences {
    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasAccessPolicy canvasAccess;
    private final CanvasCommandService canvasCommands;
    private final ResourceReadRepository resources;
    private final ResourceAccessPolicy resourceAccess;
    private final ResourceCommandService resourceCommands;
    private final MediaBindingMapper bindings;

    public void authorize(List<Reference> references, CanvasActor actor) {
        for (String id : canvasIds(references)) {
            canvasAccess.requireEdit(documents.require(id, false), actor);
        }
        for (Reference reference : references) {
            if (reference.kind().equals("BUSINESS_BINDING")) {
                String[] parts = reference.id().split(":", 3);
                resourceAccess.requireEdit(resources.require(CanvasValueType.fromCode(parts[0]), Long.parseLong(parts[1]), false), actor);
            }
        }
    }

    public void detach(String mediaId, List<Reference> references, CanvasActor actor) {
        for (String canvasId : canvasIds(references)) {
            var document = documents.require(canvasId, true);
            canvasAccess.requireEdit(document, actor);
            List<CanvasGraphOperation> operations = new ArrayList<>();
            var graph = graphs.load(document);
            var removedEdges = new HashSet<String>();
            for (var node : graph.nodes()) {
                var patch = MediaReferenceRemoval.patch(node.config(), mediaId);
                if (!patch.isEmpty()) {
                    if (mediaId.equals(node.config().path("resourceId").asText())) {
                        for (var edge : graph.edges()) {
                            if (edge.sourceNodeId().equals(node.id()) && removedEdges.add(edge.id())) {
                                operations.add(new CanvasGraphOperation.DeleteEdge(edge.id(), edge.version()));
                            }
                        }
                    }
                    operations.add(new CanvasGraphOperation.PatchNode(node.id(), node.configVersion(), null, patch));
                }
            }
            for (int start = 0; start < operations.size(); start += 500) {
                canvasCommands.edit(canvasId, new CanvasEditCommand(UUID.randomUUID().toString(),
                        operations.subList(start, Math.min(start + 500, operations.size()))), actor);
            }
        }
        var targets = bindings.selectList(Wrappers.<MediaBindingEntity>lambdaQuery()
                .eq(MediaBindingEntity::getMediaId, mediaId)
                .orderByAsc(MediaBindingEntity::getResourceType, MediaBindingEntity::getResourceId));
        for (var binding : targets) {
            var type = CanvasValueType.fromCode(binding.getResourceType());
            var header = resources.require(type, binding.getResourceId(), true);
            resourceAccess.requireEdit(header, actor);
            ObjectNode data = resources.data(header);
            String field = binding.getFieldName();
            var current = data.at("/" + field.replace('.', '/'));
            if (!current.isTextual() || !current.textValue().equals("media://" + mediaId)) {
                continue;
            }
            ObjectNode patch = data.objectNode();
            int dot = field.indexOf('.');
            if (dot < 0) {
                patch.putNull(field);
            } else {
                String root = field.substring(0, dot);
                ObjectNode nested = ((ObjectNode) data.path(root)).deepCopy();
                nested.putNull(field.substring(dot + 1));
                patch.set(root, nested);
            }
            Integer parent = type == CanvasValueType.ASSET_ITEM
                    ? resources.require(CanvasValueType.ASSET, data.path("assetId").asLong(), true).version() : null;
            resourceCommands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), type,
                    binding.getResourceId(), header.version(), patch, parent), ResourceCommandContext.user(actor, null));
        }
    }

    private List<String> canvasIds(List<Reference> references) {
        return references.stream().filter(item -> item.kind().equals("CANVAS_NODE") && item.retention().equals("LIVE"))
                .map(Reference::canvasId).distinct().sorted().toList();
    }
}
