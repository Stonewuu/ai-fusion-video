package com.stonewu.fusion.service.canvas.command;

import com.stonewu.fusion.service.canvas.CanvasGraphValidator;
import com.stonewu.fusion.service.canvas.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 命名版本恢复保留对象编号，内容回到选定快照，所有版本继续单调递增。
 */
@Component
@RequiredArgsConstructor
public class CanvasCheckpointEngine {
    private final CanvasGraphValidator validator;

    public CanvasGraphMutation restore(CanvasGraph current, CanvasGraph snapshot,
                                       Map<String, CanvasNode> deletedNodes, Map<String, CanvasEdge> deletedEdges) {
        if (current.schemaVersion() != snapshot.schemaVersion() || !Objects.equals(current.projectId(),
                snapshot.projectId())) {
            throw new CanvasException(CanvasErrorCode.UNSUPPORTED_SCHEMA, "命名版本与当前画布协议或项目不一致");
        }
        var previousNodes = current.nodes().stream().collect(Collectors.toMap(CanvasNode::id, Function.identity()));
        var nodes = new ArrayList<CanvasNode>();
        for (var target : snapshot.nodes()) {
            var previous = previousNodes.get(target.id());
            var stored = previous == null ? deletedNodes.get(target.id()) : previous;
            if (stored == null || stored.kind() != target.kind() || stored.schemaVersion() != target.schemaVersion()) {
                throw conflict(target.id());
            }
            boolean configChanged = previous == null || !stored.title().equals(target.title())
                    || !stored.config().equals(target.config()) || !Objects.equals(stored.resource(),
                    target.resource());
            boolean layoutChanged = previous == null || !stored.layout().equals(target.layout());
            nodes.add(
                    new CanvasNode(target.id(), target.kind(), target.schemaVersion(), target.title(), target.layout(),
                            target.config().deepCopy(), target.resource(),
                            Math.addExact(stored.configVersion(), configChanged ? 1 : 0),
                            Math.addExact(stored.layoutVersion(), layoutChanged ? 1 : 0)));
        }
        var previousEdges = current.edges().stream().collect(Collectors.toMap(CanvasEdge::id, Function.identity()));
        var edges = new ArrayList<CanvasEdge>();
        for (var target : snapshot.edges()) {
            var previous = previousEdges.get(target.id());
            var stored = previous == null ? deletedEdges.get(target.id()) : previous;
            if (stored == null) {
                throw conflict(target.id());
            }
            var candidate = new CanvasEdge(target.id(), target.kind(), target.sourceNodeId(), target.sourcePort(),
                    target.targetNodeId(),
                    target.targetPort(), target.inputIndex(), target.output(), stored.version());
            edges.add(previous != null && candidate.equals(previous) ? previous : new CanvasEdge(candidate.id(),
                    candidate.kind(),
                    candidate.sourceNodeId(), candidate.sourcePort(), candidate.targetNodeId(), candidate.targetPort(),
                    candidate.inputIndex(),
                    candidate.output(), Math.addExact(stored.version(), 1)));
        }
        var result = new CanvasGraph(current.schemaVersion(), current.projectId(), nodes, edges);
        validator.validate(result);
        return new CanvasGraphMutation(result, CanvasEditEngine.difference(current, result),
                CanvasEditEngine.difference(result, current));
    }

    private CanvasException conflict(String id) {
        return new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT,
                "命名版本中的对象记录已变化或被清理，无法直接恢复", Map.of("id", id));
    }
}
