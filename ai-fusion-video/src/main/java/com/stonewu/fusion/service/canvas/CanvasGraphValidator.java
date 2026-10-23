package com.stonewu.fusion.service.canvas;

import com.stonewu.fusion.service.canvas.capability.*;
import com.stonewu.fusion.service.canvas.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.regex.Pattern;

/**
 * 文档可保留缺少输入的草稿；执行所需资源与模型能力另在运行计划中校验。
 */
@Component
@RequiredArgsConstructor
public class CanvasGraphValidator {
    public static final int MAX_NODES = 10_000;
    public static final int MAX_EDGES = 30_000;
    private static final Pattern UUID = Pattern.compile(
            "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}");

    private final CanvasCapabilityRegistry registry;
    private final CanvasConfigValidator configValidator;

    public CanvasGraphValidation validate(CanvasGraph graph) {
        if (graph == null || graph.nodes() == null || graph.edges() == null) {
            throw error(CanvasErrorCode.INVALID_GRAPH, "画布必须包含节点和连线集合");
        }
        if (graph.schemaVersion() != CanvasCapabilityRegistry.SCHEMA_VERSION) {
            throw error(CanvasErrorCode.UNSUPPORTED_SCHEMA, "画布结构版本不受支持");
        }
        if (graph.projectId() != null && graph.projectId() <= 0) {
            throw error(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "项目 ID 无效");
        }
        if (graph.nodes().size() > MAX_NODES || graph.edges().size() > MAX_EDGES) {
            throw error(CanvasErrorCode.INVALID_GRAPH, "画布节点或连线数量超出限制");
        }
        Map<String, CanvasNode> nodes = indexNodes(graph);
        validateGroups(nodes);
        validateEdges(graph.edges(), nodes);
        return new CanvasGraphValidation(topologicalOrder(nodes, graph.edges()));
    }

    private Map<String, CanvasNode> indexNodes(CanvasGraph graph) {
        Map<String, CanvasNode> nodes = new LinkedHashMap<>();
        for (CanvasNode node : graph.nodes()) {
            if (node == null) {
                throw error(CanvasErrorCode.INVALID_GRAPH, "节点不能为空");
            }
            requireId(node.id());
            if (nodes.putIfAbsent(node.id(), node) != null) {
                throw error(CanvasErrorCode.INVALID_GRAPH, "节点 ID 重复: " + node.id());
            }
            CanvasNodeCapability capability = registry.require(node.kind());
            if (node.schemaVersion() != capability.schemaVersion()) {
                throw error(CanvasErrorCode.UNSUPPORTED_SCHEMA, "节点结构版本不受支持: " + node.id());
            }
            if (node.title() == null || node.title().isBlank() || node.title().length() > 200) {
                throw error(CanvasErrorCode.INVALID_GRAPH, "节点名称不能为空且不能超过 200 字符");
            }
            if (node.configVersion() < 0 || node.layoutVersion() < 0) {
                throw error(CanvasErrorCode.INVALID_GRAPH, "节点版本不能为负数");
            }
            validateLayout(node.layout());
            if (capability.requiresProject() && graph.projectId() == null) {
                throw error(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "该节点需要关联项目: " + node.id());
            }
            validateResource(node, capability);
            try {
                configValidator.validateDraft(capability, node.config());
                CanvasBatchDefinition.draft(node, registry, configValidator);
                if (node.kind() == CanvasNodeKind.WRITEBACK && node.config().has("binding")) {
                    CanvasWritebackFields.require(node.config());
                }
            } catch (CanvasException failure) {
                Map<String, Object> details = new HashMap<>(failure.getDetails());
                details.put("nodeId", node.id());
                throw new CanvasException(failure.getCode(), failure.getErrorCode(), failure.getMessage(), details);
            }
        }
        return nodes;
    }

    private void validateResource(CanvasNode node, CanvasNodeCapability capability) {
        CanvasResourceReference resource = node.resource();
        if (capability.resourceType() == null) {
            if (resource != null) {
                throw error(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "该节点不接受业务对象绑定");
            }
            return;
        }
        if (resource == null) {
            return;
        }
        if (resource.type() != capability.resourceType()
                || resource.id() == null || resource.id() <= 0
                || resource.version() == null || resource.version() < 0) {
            throw error(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "业务对象类型、ID 或版本无效: " + node.id());
        }
    }

    private void validateLayout(CanvasNodeLayout layout) {
        if (layout == null || !Double.isFinite(layout.x()) || !Double.isFinite(layout.y())) {
            throw error(CanvasErrorCode.INVALID_GRAPH, "节点坐标必须为有限数值");
        }
        if (!validSize(layout.width()) || !validSize(layout.height())) {
            throw error(CanvasErrorCode.INVALID_GRAPH, "节点宽高必须为正的有限数值");
        }
    }

    private boolean validSize(Double value) {
        return value == null || (Double.isFinite(value) && value > 0);
    }

    private void validateGroups(Map<String, CanvasNode> nodes) {
        for (CanvasNode node : nodes.values()) {
            String parentId = node.layout().parentId();
            if (parentId != null && (!nodes.containsKey(parentId) || nodes.get(parentId)
                    .kind() != CanvasNodeKind.GROUP)) {
                throw error(CanvasErrorCode.INVALID_GROUP, "父节点必须是同一画布中的分组: " + node.id());
            }
        }
        Set<String> complete = new HashSet<>();
        for (String id : nodes.keySet()) {
            Set<String> path = new HashSet<>();
            String current = id;
            while (current != null && !complete.contains(current)) {
                if (!path.add(current)) {
                    throw error(CanvasErrorCode.INVALID_GROUP, "分组关系形成循环");
                }
                current = nodes.get(current).layout().parentId();
            }
            complete.addAll(path);
        }
    }

    private void validateEdges(List<CanvasEdge> edges, Map<String, CanvasNode> nodes) {
        Set<String> ids = new HashSet<>();
        Set<String> inputSlots = new HashSet<>();
        Map<String, Integer> inputCounts = new HashMap<>();
        for (CanvasEdge edge : edges) {
            if (edge == null) {
                throw error(CanvasErrorCode.INVALID_GRAPH, "连线不能为空");
            }
            requireId(edge.id());
            if (!ids.add(edge.id())) {
                throw error(CanvasErrorCode.INVALID_GRAPH, "连线 ID 重复");
            }
            if (edge.kind() == null || edge.version() < 0
                    || !nodes.containsKey(edge.sourceNodeId()) || !nodes.containsKey(edge.targetNodeId())) {
                throw error(CanvasErrorCode.INVALID_GRAPH, "连线类型、版本或端点无效: " + edge.id());
            }
            if (edge.kind() != CanvasEdgeKind.EXECUTION) {
                if (edge.sourcePort() != null || edge.targetPort() != null || edge.inputIndex() != null || edge.output() != null) {
                    throw error(CanvasErrorCode.INVALID_PORT, "业务引用与注释连线不能携带执行绑定");
                }
                continue;
            }
            validateExecutionEdge(edge, nodes, inputSlots, inputCounts);
        }
    }

    private void validateExecutionEdge(CanvasEdge edge, Map<String, CanvasNode> nodes,
                                       Set<String> inputSlots, Map<String, Integer> inputCounts) {
        CanvasNode source = nodes.get(edge.sourceNodeId());
        CanvasNode target = nodes.get(edge.targetNodeId());
        CanvasPortDefinition output = requirePort(registry.require(source.kind()).outputs(), edge.sourcePort());
        CanvasPortDefinition input = requirePort(registry.require(target.kind()).inputs(), edge.targetPort());
        if (!input.accepts(output.resolveType(source.config()), target.config())
                || (output.collection() && !input.collection())) {
            throw error(CanvasErrorCode.INCOMPATIBLE_PORT, "连线数据类型或集合类型不匹配: " + edge.id());
        }
        if (edge.inputIndex() == null || edge.inputIndex() < 0 || (!input.collection() && edge.inputIndex() != 0)) {
            throw error(CanvasErrorCode.INVALID_PORT, "输入连接序号无效: " + edge.id());
        }
        String inputKey = target.id() + ":" + input.id();
        if (!inputSlots.add(inputKey + ":" + edge.inputIndex())) {
            throw error(CanvasErrorCode.DUPLICATE_INPUT, "同一输入位置不能重复连接");
        }
        int count = inputCounts.merge(inputKey, 1, Integer::sum);
        if (count > input.maxConnections()) {
            throw error(CanvasErrorCode.INPUT_LIMIT_EXCEEDED, "输入端口连接数量超出限制");
        }
        if (target.config().has(input.id())) {
            throw error(CanvasErrorCode.DUPLICATE_INPUT, "输入不能同时使用手填值和连线: " + input.id());
        }
        if (edge.output() != null) {
            if ((edge.output().nodeRunId() == null) == (edge.output().snapshotId() == null)) {
                throw error(CanvasErrorCode.INVALID_PORT, "固定结果必须且只能指定运行或快照");
            }
            if (edge.output().nodeRunId() != null) {
                requireId(edge.output().nodeRunId());
            }
            if (edge.output().snapshotId() != null) {
                requireId(edge.output().snapshotId());
            }
            if (edge.output().resourceId() != null) {
                requireId(edge.output().resourceId());
            }
            if (edge.output().outputId() == null || edge.output().outputId().isBlank()
                    || edge.output().outputId().length() > 128) {
                throw error(CanvasErrorCode.INVALID_PORT, "固定输出标识无效");
            }
        }
    }

    private CanvasPortDefinition requirePort(List<CanvasPortDefinition> ports, String id) {
        return ports.stream().filter(port -> port.id().equals(id)).findFirst()
                .orElseThrow(() -> error(CanvasErrorCode.INVALID_PORT, "端口不存在: " + id));
    }

    private List<String> topologicalOrder(Map<String, CanvasNode> nodes, List<CanvasEdge> edges) {
        Map<String, Integer> incoming = new LinkedHashMap<>();
        Map<String, List<String>> outgoing = new HashMap<>();
        nodes.keySet().forEach(id -> incoming.put(id, 0));
        for (CanvasEdge edge : edges) {
            if (edge.kind() != CanvasEdgeKind.EXECUTION) {
                continue;
            }
            incoming.merge(edge.targetNodeId(), 1, Integer::sum);
            outgoing.computeIfAbsent(edge.sourceNodeId(), ignored -> new ArrayList<>()).add(edge.targetNodeId());
        }
        ArrayDeque<String> ready = new ArrayDeque<>();
        incoming.forEach((id, degree) -> {
            if (degree == 0) {
                ready.add(id);
            }
        });
        List<String> result = new ArrayList<>();
        while (!ready.isEmpty()) {
            String id = ready.removeFirst();
            result.add(id);
            for (String target : outgoing.getOrDefault(id, List.of())) {
                if (incoming.merge(target, -1, Integer::sum) == 0) {
                    ready.addLast(target);
                }
            }
        }
        if (result.size() != nodes.size()) {
            throw error(CanvasErrorCode.EXECUTION_CYCLE, "执行连线不能形成循环");
        }
        return result;
    }

    private void requireId(String value) {
        if (value == null || !UUID.matcher(value).matches()) {
            throw error(CanvasErrorCode.INVALID_GRAPH, "画布对象 ID 必须为 UUID");
        }
    }

    private CanvasException error(CanvasErrorCode code, String message) {
        return new CanvasException(code, message);
    }
}
