package com.stonewu.fusion.service.canvas.command;

import com.stonewu.fusion.service.canvas.CanvasGraphValidator;
import com.stonewu.fusion.service.canvas.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 撤销也是新命令：检查原命令的后值，恢复内容但绝不回退版本号。
 */
@Component
@RequiredArgsConstructor
public class CanvasUndoEngine {
    private final CanvasGraphValidator validator;

    public CanvasGraphMutation apply(CanvasGraph graph, CanvasGraphChangeSet after, CanvasGraphChangeSet before,
                                     Map<String, CanvasNode> deletedNodes, Map<String, CanvasEdge> deletedEdges) {
        if (after.upsertedNodes().isEmpty() && after.deletedNodeIds().isEmpty()
                && after.upsertedEdges().isEmpty() && after.deletedEdgeIds().isEmpty()) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "该操作没有可撤销的图修改");
        }
        Map<String, CanvasNode> nodes = new LinkedHashMap<>();
        graph.nodes().forEach(node -> nodes.put(node.id(), node));
        Map<String, CanvasEdge> edges = new LinkedHashMap<>();
        graph.edges().forEach(edge -> edges.put(edge.id(), edge));
        Map<String, CanvasNode> oldNodes = new LinkedHashMap<>();
        before.upsertedNodes().forEach(node -> oldNodes.put(node.id(), node));
        Map<String, CanvasEdge> oldEdges = new LinkedHashMap<>();
        before.upsertedEdges().forEach(edge -> oldEdges.put(edge.id(), edge));
        for (CanvasNode expected : after.upsertedNodes()) {
            requireSame(expected.id(), expected, nodes.get(expected.id()));
            CanvasNode previous = oldNodes.get(expected.id());
            if (previous == null) {
                nodes.remove(expected.id());
            } else {
                nodes.put(expected.id(), withVersions(previous,
                        expected.configVersion() + (previous.configVersion() == expected.configVersion() ? 0 : 1),
                        expected.layoutVersion() + (previous.layoutVersion() == expected.layoutVersion() ? 0 : 1)));
            }
        }
        for (String id : after.deletedNodeIds()) {
            CanvasNode previous = Objects.requireNonNull(oldNodes.get(id), "撤销记录缺少节点前值");
            requireSame(id, null, nodes.get(id));
            requireSame(id, withVersions(previous, previous.configVersion() + 1, previous.layoutVersion() + 1),
                    deletedNodes.get(id));
            nodes.put(id, withVersions(previous, previous.configVersion() + 2, previous.layoutVersion() + 2));
        }
        for (CanvasEdge expected : after.upsertedEdges()) {
            requireSame(expected.id(), expected, edges.get(expected.id()));
            CanvasEdge previous = oldEdges.get(expected.id());
            if (previous == null) {
                edges.remove(expected.id());
            } else {
                edges.put(expected.id(), withVersion(previous, expected.version() + 1));
            }
        }
        for (String id : after.deletedEdgeIds()) {
            CanvasEdge previous = Objects.requireNonNull(oldEdges.get(id), "撤销记录缺少连线前值");
            requireSame(id, null, edges.get(id));
            requireSame(id, withVersion(previous, previous.version() + 1), deletedEdges.get(id));
            edges.put(id, withVersion(previous, previous.version() + 2));
        }
        var result = new CanvasGraph(graph.schemaVersion(), graph.projectId(), List.copyOf(nodes.values()),
                List.copyOf(edges.values()));
        try {
            validator.validate(result);
        } catch (CanvasException conflict) {
            throw new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT,
                    "后续图修改与此次撤销冲突：" + conflict.getMessage(), conflict.getDetails());
        }
        return new CanvasGraphMutation(result, CanvasEditEngine.difference(graph, result),
                CanvasEditEngine.difference(result, graph));
    }

    private CanvasNode withVersions(CanvasNode node, long config, long layout) {
        if (config < 0 || layout < 0) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "节点版本超出上限");
        }
        return new CanvasNode(node.id(), node.kind(), node.schemaVersion(), node.title(), node.layout(),
                node.config().deepCopy(),
                node.resource(), config, layout);
    }

    private CanvasEdge withVersion(CanvasEdge edge, long version) {
        if (version < 0) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "连线版本超出上限");
        }
        return new CanvasEdge(edge.id(), edge.kind(), edge.sourceNodeId(), edge.sourcePort(), edge.targetNodeId(),
                edge.targetPort(),
                edge.inputIndex(), edge.output(), version);
    }

    private void requireSame(String id, Object expected, Object actual) {
        if (!Objects.equals(expected, actual)) {
            throw new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT,
                    "对象在该操作之后已发生变化，不能直接撤销", Map.of("id", id));
        }
    }
}
