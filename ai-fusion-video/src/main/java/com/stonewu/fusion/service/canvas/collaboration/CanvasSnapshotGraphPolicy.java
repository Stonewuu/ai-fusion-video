package com.stonewu.fusion.service.canvas.collaboration;

import com.stonewu.fusion.service.canvas.CanvasGraphMutationPolicy;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.command.CanvasGraphMutation;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

@Component
@RequiredArgsConstructor
public class CanvasSnapshotGraphPolicy implements CanvasGraphMutationPolicy {
    private final CanvasPinnedOutputs outputs;
    private final CanvasValueSnapshots snapshots;
    private final CanvasCapabilityRegistry capabilities;

    @Override
    public void validate(CanvasDocument document, CanvasGraph before, CanvasGraphMutation mutation, CanvasActor actor) {
        var previousNodes = before.nodes().stream().collect(Collectors.toMap(node -> node.id(), Function.identity()));
        var previousEdges = before.edges().stream().collect(Collectors.toMap(edge -> edge.id(), Function.identity()));
        var changedSources = new HashSet<String>();
        mutation.changes().upsertedNodes().forEach(node -> {
            var old = previousNodes.get(node.id());
            if (old == null || old.kind() != node.kind() || !old.config().equals(node.config())) {
                changedSources.add(node.id());
            }
        });
        var changedEdges = mutation.changes().upsertedEdges().stream().map(edge -> edge.id())
                .collect(Collectors.toSet());
        for (var edge : mutation.graph().edges()) {
            if (edge.output() == null || edge.output().snapshotId() == null) {
                continue;
            }
            if (!changedEdges.contains(edge.id()) && !changedSources.contains(edge.sourceNodeId())) {
                continue;
            }
            var source = mutation.graph().nodes().stream().filter(node -> node.id().equals(edge.sourceNodeId()))
                    .findFirst().orElseThrow();
            var port = capabilities.require(source.kind()).outputs().stream()
                    .filter(value -> value.id().equals(edge.sourcePort())).findFirst().orElseThrow();
            var old = previousEdges.get(edge.id());
            var value = old != null && Objects.equals(old.output(), edge.output()) && old.sourceNodeId()
                    .equals(edge.sourceNodeId())
                    && old.sourcePort().equals(edge.sourcePort()) ? snapshots.get(edge.output().snapshotId()).value()
                    : outputs.resolve(document.id(), edge, actor);
            if (value.type() != port.resolveType(source.config())) {
                throw CanvasRunException.invalid("固定快照的类型与当前输出端口不一致");
            }
        }
    }
}
