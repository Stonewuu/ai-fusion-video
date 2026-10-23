package com.stonewu.fusion.service.canvas.agent;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasCommandService;
import com.stonewu.fusion.service.canvas.command.CanvasEditCommand;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation;
import com.stonewu.fusion.service.canvas.model.CanvasEdgeKind;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeLayout;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class CanvasAgentLayout {
    private final CanvasAgentAccess access;
    private final CanvasCommandService commands;

    private record Point(double x, double y) {
    }

    public Object arrange(List<String> ids, String mode, long revision, ToolExecutionContext tool) {
        if (ids.isEmpty() || ids.size() > 200 || new HashSet<>(ids).size() != ids.size()) {
            throw new BusinessException(400, "请指定 1 至 200 个不同的节点");
        }
        if (!Set.of("arrange", "left", "top", "horizontal", "vertical").contains(mode)) {
            throw new BusinessException(400, "排列方式无效");
        }
        var session = access.require(tool);
        var intent = Map.of("operation", "arrange", "version", 1, "nodeIds", ids, "mode", mode, "expectedRevision",
                revision);
        var receipt = commands.derive(session.scope().canvasId(), session.requestId(), intent, session.context(),
                (document, graph) -> {
                    if (document.revision() != revision) {
                        throw new BusinessException(409, "画布已改变，请重新读取后整理");
                    }
                    return new CanvasEditCommand(session.requestId(), layout(graph, ids, mode));
                });
        return Map.of("status", "success", "commandId", receipt.commandId(), "canvasRevision", receipt.revision(),
                "receipt", receipt);
    }

    List<CanvasGraphOperation> layout(CanvasGraph graph, List<String> ids, String mode) {
        Map<String, CanvasNode> all = graph.nodes().stream()
                .collect(Collectors.toMap(CanvasNode::id, Function.identity()));
        if (!all.keySet().containsAll(ids)) {
            throw new BusinessException(404, "选区包含不存在的节点");
        }
        Set<String> selected = Set.copyOf(ids);
        var nodes = ids.stream().map(all::get)
                .filter(node -> !node.layout().locked() && !selectedParent(node, selected, all)).toList();
        if (nodes.isEmpty()) {
            throw new BusinessException(409, "选区中没有可移动节点，锁定节点会保持原位");
        }
        Map<String, Point> original = new HashMap<>();
        nodes.forEach(node -> original.put(node.id(), world(node, all)));
        double left = original.values().stream().mapToDouble(Point::x).min().orElseThrow();
        double top = original.values().stream().mapToDouble(Point::y).min().orElseThrow();
        Map<String, Point> points = mode.equals("arrange") ? paths(graph, nodes, all, original, left,
                top) : new LinkedHashMap<>();
        if (!mode.equals("arrange")) {
            var ordered = nodes.stream().sorted(Comparator.comparingDouble(
                            node -> mode.equals("vertical") ? original.get(node.id()).y() : original.get(node.id()).x()))
                    .toList();
            double cursor = 0;
            for (var node : ordered) {
                var point = original.get(node.id());
                points.put(node.id(), switch (mode) {
                    case "left" -> new Point(left, point.y());
                    case "top" -> new Point(point.x(), top);
                    case "horizontal" -> new Point(left + cursor, top);
                    case "vertical" -> new Point(left, top + cursor);
                    default -> throw new IllegalStateException("排列方式无效");
                });
                cursor += (mode.equals("vertical") ? height(node) : width(node)) + 64;
            }
        }
        List<CanvasGraphOperation> operations = new ArrayList<>();
        for (var node : nodes) {
            var before = node.layout();
            var point = points.get(node.id());
            var parent = before.parentId() == null ? new Point(0, 0) : world(all.get(before.parentId()), all);
            operations.add(new CanvasGraphOperation.LayoutNode(node.id(), node.layoutVersion(),
                    new CanvasNodeLayout(point.x() - parent.x(), point.y() - parent.y(),
                            before.width(), before.height(), before.parentId(), before.locked(), before.collapsed())));
        }
        return operations;
    }

    private Map<String, Point> paths(CanvasGraph graph, List<CanvasNode> nodes, Map<String, CanvasNode> all,
                                     Map<String, Point> original, double left, double top) {
        var byId = nodes.stream().collect(Collectors.toMap(CanvasNode::id, Function.identity()));
        Map<String, Set<String>> outgoing = new HashMap<>(), incoming = new HashMap<>();
        nodes.forEach(node -> {
            outgoing.put(node.id(), new LinkedHashSet<>());
            incoming.put(node.id(), new LinkedHashSet<>());
        });
        for (var edge : graph.edges()) {
            if (edge.kind() != CanvasEdgeKind.EXECUTION) {
                continue;
            }
            String source = endpoint(edge.sourceNodeId(), byId.keySet(), all), target = endpoint(edge.targetNodeId(),
                    byId.keySet(), all);
            if (source == null || target == null || source.equals(target)) {
                continue;
            }
            outgoing.get(source).add(target);
            incoming.get(target).add(source);
        }
        // 执行关系先确定方向，来源和注释关系只补充不成环的布局约束。
        for (var edge : graph.edges()) {
            if (edge.kind() == CanvasEdgeKind.EXECUTION) {
                continue;
            }
            String source = endpoint(edge.sourceNodeId(), byId.keySet(), all), target = endpoint(edge.targetNodeId(),
                    byId.keySet(), all);
            if (source == null || target == null || source.equals(target) || outgoing.get(source).contains(target)) {
                continue;
            }
            var pending = new ArrayList<String>();
            pending.add(target);
            var visited = new HashSet<String>();
            while (!pending.isEmpty()) {
                String id = pending.removeLast();
                if (!visited.add(id)) {
                    continue;
                }
                pending.addAll(outgoing.get(id));
            }
            if (visited.contains(source)) {
                continue;
            }
            outgoing.get(source).add(target);
            incoming.get(target).add(source);
        }
        Comparator<String> order = Comparator.comparingDouble((String id) -> original.get(id).y())
                .thenComparingDouble(id -> original.get(id).x()).thenComparing(Comparator.naturalOrder());
        Map<String, Integer> degree = new HashMap<>(), levels = new HashMap<>(), distance = new HashMap<>();
        nodes.forEach(node -> {
            degree.put(node.id(), incoming.get(node.id()).size());
            levels.put(node.id(), 0);
            distance.put(node.id(), 0);
        });
        var ready = new ArrayList<>(byId.keySet().stream().filter(id -> degree.get(id) == 0).sorted(order).toList());
        var sorted = new ArrayList<String>();
        while (!ready.isEmpty()) {
            String id = ready.removeFirst();
            sorted.add(id);
            for (String next : outgoing.get(id)) {
                levels.put(next, Math.max(levels.get(next), levels.get(id) + 1));
                degree.put(next, degree.get(next) - 1);
                if (degree.get(next) == 0) {
                    ready.add(next);
                    ready.sort(order);
                }
            }
        }
        if (sorted.size() != nodes.size()) {
            throw new BusinessException(409, "选区的连线跨分组形成循环，请选择具体节点后整理");
        }
        for (String id : sorted.reversed()) {
            for (String next : outgoing.get(id)) {
                distance.put(id, Math.max(distance.get(id), distance.get(next) + 1));
            }
        }
        var remaining = new HashSet<>(sorted);
        var components = new ArrayList<List<String>>();
        for (String seed : sorted.stream().sorted(order).toList()) {
            if (!remaining.remove(seed)) {
                continue;
            }
            var component = new ArrayList<String>();
            component.add(seed);
            for (int index = 0; index < component.size(); index++) {
                var neighbors = new HashSet<>(outgoing.get(component.get(index)));
                neighbors.addAll(incoming.get(component.get(index)));
                for (String neighbor : neighbors.stream().sorted(order).toList()) {
                    if (remaining.remove(neighbor)) {
                        component.add(neighbor);
                    }
                }
            }
            components.add(component);
        }
        Map<String, Point> result = new HashMap<>();
        double rowTop = top;
        for (var component : components) {
            if (component.size() == 1) {
                continue;
            }
            Map<Integer, Double> widths = new TreeMap<>(), columns = new HashMap<>();
            for (String id : component) {
                widths.merge(levels.get(id), width(byId.get(id)), Math::max);
            }
            double column = left;
            for (var entry : widths.entrySet()) {
                columns.put(entry.getKey(), column);
                column += entry.getValue() + 96;
            }
            var unused = new HashSet<>(component);
            while (!unused.isEmpty()) {
                String current = unused.stream().min(Comparator.comparingInt((String id) -> levels.get(id))
                        .thenComparing(Comparator.comparingInt((String id) -> distance.get(id)).reversed())
                        .thenComparing(order)).orElseThrow();
                var chain = new ArrayList<String>();
                while (current != null) {
                    chain.add(current);
                    unused.remove(current);
                    current = outgoing.get(current).stream().filter(unused::contains)
                            .min(Comparator.comparingInt((String id) -> distance.get(id)).reversed()
                                    .thenComparing(order)).orElse(null);
                }
                double rowHeight = chain.stream().mapToDouble(id -> height(byId.get(id))).max().orElseThrow();
                for (String id : chain) {
                    result.put(id,
                            new Point(columns.get(levels.get(id)), rowTop + (rowHeight - height(byId.get(id))) / 2));
                }
                rowTop += rowHeight + 80;
            }
            rowTop += 32;
        }
        var loose = components.stream().filter(item -> item.size() == 1).map(List::getFirst).toList();
        int count = (int) Math.ceil(Math.sqrt(loose.size()));
        double x = left, rowHeight = 0;
        for (int index = 0; index < loose.size(); index++) {
            if (index > 0 && index % count == 0) {
                rowTop += rowHeight + 80;
                x = left;
                rowHeight = 0;
            }
            String id = loose.get(index);
            result.put(id, new Point(x, rowTop));
            x += width(byId.get(id)) + 96;
            rowHeight = Math.max(rowHeight, height(byId.get(id)));
        }
        return result;
    }

    private String endpoint(String id, Set<String> selection, Map<String, CanvasNode> all) {
        while (id != null) {
            if (selection.contains(id)) {
                return id;
            }
            id = all.get(id).layout().parentId();
        }
        return null;
    }

    private boolean selectedParent(CanvasNode node, Set<String> selected, Map<String, CanvasNode> all) {
        return node.layout().parentId() != null && endpoint(node.layout().parentId(), selected, all) != null;
    }

    private Point world(CanvasNode node, Map<String, CanvasNode> all) {
        double x = 0, y = 0;
        while (node != null) {
            x += node.layout().x();
            y += node.layout().y();
            node = node.layout().parentId() == null ? null : all.get(node.layout().parentId());
        }
        return new Point(x, y);
    }

    private double width(CanvasNode node) {
        return node.layout().width() == null ? 300 : node.layout().width();
    }

    private double height(CanvasNode node) {
        return node.layout().height() == null ? 280 : node.layout().height();
    }
}
