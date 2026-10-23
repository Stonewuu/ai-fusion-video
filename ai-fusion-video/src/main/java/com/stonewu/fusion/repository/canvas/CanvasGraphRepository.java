package com.stonewu.fusion.repository.canvas;

import com.stonewu.fusion.mapper.canvas.CanvasNodeMapper;
import com.stonewu.fusion.mapper.canvas.CanvasEdgeMapper;
import com.stonewu.fusion.entity.canvas.CanvasNodeEntity;
import com.stonewu.fusion.entity.canvas.CanvasEdgeEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import java.util.stream.Collectors;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.command.CanvasGraphMutation;
import com.stonewu.fusion.service.canvas.model.*;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Repository;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 调用方在同一事务内持有画布行锁，所有修改仍检查对象版本与所属画布。
 */
@Repository
@RequiredArgsConstructor
public class CanvasGraphRepository {
    private final CanvasNodeMapper nodes;
    private final CanvasEdgeMapper edges;
    private final CanvasJson json;

    public CanvasGraph load(CanvasDocument document) {
        List<CanvasNode> loadedNodes = nodes.ordered(document.id()).stream().map(this::node).toList();
        List<CanvasEdge> loadedEdges = edges.ordered(document.id()).stream().map(this::edge).toList();
        return new CanvasGraph(document.schemaVersion(), document.projectId(), loadedNodes, loadedEdges);
    }

    public void save(String canvasId, CanvasGraphMutation mutation) {
        save(canvasId, mutation, false);
    }

    public void restore(String canvasId, CanvasGraphMutation mutation) {
        save(canvasId, mutation, true);
    }

    public Map<String, CanvasNode> deletedNodes(String canvasId, List<String> ids) {
        if (ids.isEmpty()) {
            return Map.of();
        }
        return nodes.deleted(canvasId, ids).stream().map(this::node)
                .collect(Collectors.toMap(CanvasNode::id, value -> value));
    }

    public Map<String, CanvasEdge> deletedEdges(String canvasId, List<String> ids) {
        if (ids.isEmpty()) {
            return Map.of();
        }
        return edges.deleted(canvasId, ids).stream().map(this::edge)
                .collect(Collectors.toMap(CanvasEdge::id, value -> value));
    }

    private void save(String canvasId, CanvasGraphMutation mutation, boolean restoring) {
        Map<String, CanvasNode> previousNodes = new HashMap<>();
        mutation.previousValues().upsertedNodes().forEach(node -> previousNodes.put(node.id(), node));
        Map<String, CanvasEdge> previousEdges = new HashMap<>();
        mutation.previousValues().upsertedEdges().forEach(edge -> previousEdges.put(edge.id(), edge));

        // 先为新节点建立记录，再设置分组，允许同一命令按任意顺序创建父子节点。
        for (CanvasNode node : mutation.changes().upsertedNodes()) {
            if (!previousNodes.containsKey(node.id())) {
                if (restoring) {
                    restoreNode(canvasId, node);
                } else {
                    insertNode(canvasId, node);
                }
            }
        }
        for (CanvasNode node : mutation.changes().upsertedNodes()) {
            CanvasNode previous = previousNodes.get(node.id());
            if (previous != null) {
                updateNode(canvasId, previous, node);
            } else if (node.layout().parentId() != null) {
                requireUpdated(
                        nodes.update(Wrappers.<CanvasNodeEntity>lambdaUpdate().eq(CanvasNodeEntity::getId, node.id())
                                .eq(CanvasNodeEntity::getCanvasId, canvasId)
                                .set(CanvasNodeEntity::getParentId, node.layout().parentId())));
            }
        }

        // 在事务内释放受影响的输入位置，然后写最终连线，支持交换两个参考输入的顺序。
        for (CanvasEdge previous : previousEdges.values()) {
            requireUpdated(edges.update(Wrappers.<CanvasEdgeEntity>lambdaUpdate()
                    .eq(CanvasEdgeEntity::getId, previous.id()).eq(CanvasEdgeEntity::getCanvasId, canvasId)
                    .eq(CanvasEdgeEntity::getVersion, previous.version()).eq(CanvasEdgeEntity::getDeleted, false)
                    .set(CanvasEdgeEntity::getDeleted, true)));
        }
        for (CanvasEdge edge : mutation.changes().upsertedEdges()) {
            CanvasEdge previous = previousEdges.get(edge.id());
            if (previous == null && restoring) {
                var tombstone = new CanvasEdge(edge.id(), edge.kind(), edge.sourceNodeId(), edge.sourcePort(),
                        edge.targetNodeId(),
                        edge.targetPort(), edge.inputIndex(), edge.output(), edge.version() - 1);
                updateEdge(canvasId, tombstone, edge);
            } else if (previous == null) {
                insertEdge(canvasId, edge);
            } else {
                updateEdge(canvasId, previous, edge);
            }
        }
        for (String id : mutation.changes().deletedEdgeIds()) {
            CanvasEdge previous = previousEdges.get(id);
            requireUpdated(edges.deleteGraphEdge(id, canvasId, previous.version()));
        }
        for (String id : mutation.changes().deletedNodeIds()) {
            CanvasNode previous = previousNodes.get(id);
            requireUpdated(nodes.deleteGraphNode(id, canvasId, previous.configVersion(), previous.layoutVersion()));
        }
    }

    private void restoreNode(String canvasId, CanvasNode node) {
        requireUpdated(nodes.restoreGraph(nodeRow(canvasId, node), node.configVersion() - 1, node.layoutVersion() - 1));
    }

    private void insertNode(String canvasId, CanvasNode node) {
        nodes.insertGraph(nodeRow(canvasId, node));
    }

    private void updateNode(String canvasId, CanvasNode previous, CanvasNode node) {
        requireUpdated(nodes.updateGraph(nodeRow(canvasId, node), previous.configVersion(), previous.layoutVersion()));
    }

    private void insertEdge(String canvasId, CanvasEdge edge) {
        edges.insertGraph(edgeRow(canvasId, edge));
    }

    private void updateEdge(String canvasId, CanvasEdge previous, CanvasEdge edge) {
        requireUpdated(edges.updateGraph(edgeRow(canvasId, edge), previous.version()));
    }

    private CanvasNode node(CanvasNodeEntity row) {
        String type = row.getResourceType();
        CanvasResourceReference resource = type == null ? null : new CanvasResourceReference(
                CanvasValueType.fromCode(type),
                row.getResourceId(), row.getResourceVersion());
        CanvasNodeLayout layout = new CanvasNodeLayout(row.getX(), row.getY(),
                row.getWidth(),
                row.getHeight(), row.getParentId(), row.getLocked(),
                row.getCollapsed());
        return new CanvasNode(row.getId(), CanvasNodeKind.fromCode(row.getKind()),
                row.getSchemaVersion(),
                row.getTitle(), layout, json.read(row.getConfigJson(), ObjectNode.class), resource,
                row.getConfigVersion(), row.getLayoutVersion());
    }

    private CanvasEdge edge(CanvasEdgeEntity row) {
        String output = row.getOutputJson();
        return new CanvasEdge(row.getId(), CanvasEdgeKind.fromCode(row.getKind()),
                row.getSourceNodeId(),
                row.getSourcePort(), row.getTargetNodeId(), row.getTargetPort(),
                row.getInputIndex(),
                output == null ? null : json.read(output, CanvasOutputReference.class), row.getVersion());
    }

    private void requireUpdated(int count) {
        if (count != 1) {
            throw new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT, "画布对象已被修改或删除", Map.of());
        }
    }

    private CanvasNodeEntity nodeRow(String canvasId, CanvasNode node) {
        var ref = node.resource();
        var layout = node.layout();
        return CanvasNodeEntity.builder().id(node.id()).canvasId(canvasId).kind(node.kind().code())
                .schemaVersion(node.schemaVersion()).title(node.title()).x(layout.x()).y(layout.y())
                .width(layout.width()).height(layout.height()).parentId(layout.parentId()).locked(layout.locked())
                .collapsed(layout.collapsed()).configJson(json.write(node.config()))
                .resourceType(ref == null ? null : ref.type().code()).resourceId(ref == null ? null : ref.id())
                .resourceVersion(ref == null ? null : ref.version()).configVersion(node.configVersion())
                .layoutVersion(node.layoutVersion()).build();
    }

    private CanvasEdgeEntity edgeRow(String canvasId, CanvasEdge edge) {
        return CanvasEdgeEntity.builder().id(edge.id()).canvasId(canvasId).kind(edge.kind().code())
                .sourceNodeId(edge.sourceNodeId()).sourcePort(edge.sourcePort()).targetNodeId(edge.targetNodeId())
                .targetPort(edge.targetPort()).inputIndex(edge.inputIndex())
                .outputJson(edge.output() == null ? null : json.write(edge.output())).version(edge.version()).build();
    }
}
