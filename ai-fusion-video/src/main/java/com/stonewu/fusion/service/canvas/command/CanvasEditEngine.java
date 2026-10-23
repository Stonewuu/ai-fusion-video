package com.stonewu.fusion.service.canvas.command;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasGraphValidator;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation.*;
import com.stonewu.fusion.service.canvas.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * 纯图编辑逻辑；权限、幂等及事务提交由命令服务处理。
 */
@Component
@RequiredArgsConstructor
public class CanvasEditEngine {
    private final CanvasGraphValidator validator;

    public CanvasGraphMutation apply(CanvasGraph original, List<CanvasGraphOperation> operations) {
        validator.validate(original);
        if (operations == null || operations.isEmpty() || operations.size() > 500) {
            throw invalid("每次提交必须包含 1 到 500 个编辑操作");
        }
        Map<String, CanvasNode> nodes = new LinkedHashMap<>();
        original.nodes().forEach(node -> nodes.put(node.id(), copy(node)));
        Map<String, CanvasEdge> edges = new LinkedHashMap<>();
        original.edges().forEach(edge -> edges.put(edge.id(), edge));
        for (CanvasGraphOperation operation : operations) {
            if (operation == null) {
                throw invalid("编辑操作不能为空");
            }
            switch (operation) {
                case AssertNode command -> {
                    CanvasNode node = requireNode(nodes, command.nodeId());
                    requireVersion(node.id(), "configVersion", command.expectedConfigVersion(), node.configVersion());
                }
                case CreateNode command -> createNode(nodes, command);
                case PatchNode command -> patchNode(nodes, command);
                case LayoutNode command -> layoutNode(nodes, command);
                case BindResource command -> bindResource(nodes, command);
                case DeleteNode command -> deleteNode(nodes, edges, command);
                case CreateEdge command -> createEdge(edges, command);
                case ReplaceEdge command -> replaceEdge(edges, command);
                case DeleteEdge command -> deleteEdge(edges, command);
            }
        }
        CanvasGraph result = new CanvasGraph(original.schemaVersion(), original.projectId(),
                List.copyOf(nodes.values()), List.copyOf(edges.values()));
        validator.validate(result);
        return new CanvasGraphMutation(result, difference(original, result), difference(result, original));
    }

    private void createNode(Map<String, CanvasNode> nodes, CreateNode command) {
        CanvasNode node = command.node();
        if (node == null || node.configVersion() != 0 || node.layoutVersion() != 0) {
            throw invalid("新节点必须从零版本创建");
        }
        if (nodes.putIfAbsent(node.id(), copy(node)) != null) {
            throw invalid("节点已存在: " + node.id());
        }
    }

    private void patchNode(Map<String, CanvasNode> nodes, PatchNode command) {
        CanvasNode node = requireNode(nodes, command.nodeId());
        requireVersion(node.id(), "configVersion", command.expectedConfigVersion(), node.configVersion());
        if (command.title() == null && (command.configPatch() == null || command.configPatch().isEmpty())) {
            throw invalid("修改节点必须指定名称或配置字段");
        }
        ObjectNode config = node.config().deepCopy();
        if (command.configPatch() != null) {
            command.configPatch().fields().forEachRemaining(entry -> {
                JsonNode value = entry.getValue();
                if (value.isNull()) {
                    config.remove(entry.getKey());
                } else {
                    config.set(entry.getKey(), value.deepCopy());
                }
            });
        }
        nodes.put(node.id(), new CanvasNode(node.id(), node.kind(), node.schemaVersion(),
                command.title() == null ? node.title() : command.title(), node.layout(), config,
                node.resource(), increment(node.configVersion()), node.layoutVersion()));
    }

    private void layoutNode(Map<String, CanvasNode> nodes, LayoutNode command) {
        CanvasNode node = requireNode(nodes, command.nodeId());
        requireVersion(node.id(), "layoutVersion", command.expectedLayoutVersion(), node.layoutVersion());
        CanvasNodeLayout layout = command.layout();
        if (layout == null) {
            throw invalid("必须指定节点布局");
        }
        boolean changesPosition = node.layout().x() != layout.x() || node.layout().y() != layout.y()
                || !Objects.equals(node.layout().parentId(), layout.parentId());
        if (node.layout().locked() && changesPosition) {
            throw invalid("节点位置已锁定，请先解锁");
        }
        nodes.put(node.id(), new CanvasNode(node.id(), node.kind(), node.schemaVersion(), node.title(), layout,
                node.config(), node.resource(), node.configVersion(), increment(node.layoutVersion())));
    }

    private void bindResource(Map<String, CanvasNode> nodes, BindResource command) {
        CanvasNode node = requireNode(nodes, command.nodeId());
        requireVersion(node.id(), "configVersion", command.expectedConfigVersion(), node.configVersion());
        nodes.put(node.id(), new CanvasNode(node.id(), node.kind(), node.schemaVersion(), node.title(), node.layout(),
                node.config(), command.resource(), increment(node.configVersion()), node.layoutVersion()));
    }

    private void deleteNode(Map<String, CanvasNode> nodes, Map<String, CanvasEdge> edges, DeleteNode command) {
        CanvasNode node = requireNode(nodes, command.nodeId());
        requireVersion(node.id(), "configVersion", command.expectedConfigVersion(), node.configVersion());
        requireVersion(node.id(), "layoutVersion", command.expectedLayoutVersion(), node.layoutVersion());
        nodes.remove(node.id());
        for (CanvasNode child : new ArrayList<>(nodes.values())) {
            if (!Objects.equals(child.layout().parentId(), node.id())) {
                continue;
            }
            CanvasNodeLayout old = child.layout();
            CanvasNodeLayout detached = new CanvasNodeLayout(old.x() + node.layout().x(), old.y() + node.layout().y(),
                    old.width(), old.height(), node.layout().parentId(), old.locked(), old.collapsed());
            nodes.put(child.id(),
                    new CanvasNode(child.id(), child.kind(), child.schemaVersion(), child.title(), detached,
                            child.config(), child.resource(), child.configVersion(), increment(child.layoutVersion())));
        }
        edges.values().removeIf(edge -> edge.sourceNodeId().equals(node.id()) || edge.targetNodeId().equals(node.id()));
    }

    private void createEdge(Map<String, CanvasEdge> edges, CreateEdge command) {
        CanvasEdge edge = command.edge();
        if (edge == null || edge.version() != 0) {
            throw invalid("新连线必须从零版本创建");
        }
        if (edges.putIfAbsent(edge.id(), edge) != null) {
            throw invalid("连线已存在: " + edge.id());
        }
    }

    private void replaceEdge(Map<String, CanvasEdge> edges, ReplaceEdge command) {
        CanvasEdge edge = command.edge();
        if (edge == null) {
            throw invalid("必须指定连线");
        }
        CanvasEdge previous = requireEdge(edges, edge.id());
        requireVersion(edge.id(), "version", command.expectedVersion(), previous.version());
        if (edge.version() != command.expectedVersion()) {
            throw invalid("连线内容版本与前置版本不一致");
        }
        edges.put(edge.id(),
                new CanvasEdge(edge.id(), edge.kind(), edge.sourceNodeId(), edge.sourcePort(), edge.targetNodeId(),
                        edge.targetPort(), edge.inputIndex(), edge.output(), increment(previous.version())));
    }

    private void deleteEdge(Map<String, CanvasEdge> edges, DeleteEdge command) {
        CanvasEdge edge = requireEdge(edges, command.edgeId());
        requireVersion(edge.id(), "version", command.expectedVersion(), edge.version());
        edges.remove(edge.id());
    }

    private CanvasNode requireNode(Map<String, CanvasNode> nodes, String id) {
        CanvasNode result = nodes.get(id);
        if (result == null) {
            throw new CanvasException(409, CanvasErrorCode.RESOURCE_DELETED,
                    "节点不存在或已移除", Map.of("nodeId", Objects.toString(id, "")));
        }
        return result;
    }

    private CanvasEdge requireEdge(Map<String, CanvasEdge> edges, String id) {
        CanvasEdge result = edges.get(id);
        if (result == null) {
            throw new CanvasException(409, CanvasErrorCode.RESOURCE_DELETED,
                    "连线不存在或已移除", Map.of("edgeId", Objects.toString(id, "")));
        }
        return result;
    }

    private void requireVersion(String id, String field, long expected, long actual) {
        if (expected != actual) {
            throw new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT,
                    "内容已被其他操作修改，请读取最新版本",
                    Map.of("id", id, "field", field, "expected", expected, "actual", actual));
        }
    }

    private long increment(long value) {
        if (value == Long.MAX_VALUE) {
            throw invalid("版本号已达到上限");
        }
        return value + 1;
    }

    private CanvasNode copy(CanvasNode node) {
        return new CanvasNode(node.id(), node.kind(), node.schemaVersion(), node.title(), node.layout(),
                node.config() == null ? null : node.config().deepCopy(), node.resource(), node.configVersion(),
                node.layoutVersion());
    }

    public static CanvasGraphChangeSet difference(CanvasGraph before, CanvasGraph after) {
        Map<String, CanvasNode> oldNodes = new LinkedHashMap<>();
        before.nodes().forEach(node -> oldNodes.put(node.id(), node));
        List<CanvasNode> upsertedNodes = new ArrayList<>();
        for (CanvasNode node : after.nodes()) {
            if (!node.equals(oldNodes.remove(node.id()))) {
                upsertedNodes.add(node);
            }
        }
        Map<String, CanvasEdge> oldEdges = new LinkedHashMap<>();
        before.edges().forEach(edge -> oldEdges.put(edge.id(), edge));
        List<CanvasEdge> upsertedEdges = new ArrayList<>();
        for (CanvasEdge edge : after.edges()) {
            if (!edge.equals(oldEdges.remove(edge.id()))) {
                upsertedEdges.add(edge);
            }
        }
        return new CanvasGraphChangeSet(upsertedNodes, List.copyOf(oldNodes.keySet()), upsertedEdges,
                List.copyOf(oldEdges.keySet()));
    }

    private CanvasException invalid(String message) {
        return new CanvasException(CanvasErrorCode.INVALID_COMMAND, message);
    }
}
