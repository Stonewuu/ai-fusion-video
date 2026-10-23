package com.stonewu.fusion.service.canvas.collaboration;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository;
import com.stonewu.fusion.service.canvas.*;
import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import com.stonewu.fusion.service.canvas.command.CanvasCreateCommand;
import com.stonewu.fusion.service.canvas.command.CanvasEditEngine;
import com.stonewu.fusion.service.canvas.command.CanvasGraphMutation;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.resource.model.ProjectForkCommand;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.UUID;

@Component
@RequiredArgsConstructor
public class CanvasProjectForkWriter {
    private final CanvasCommandService commands;
    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasJournalRepository journal;
    private final CanvasGraphValidator validator;
    private final CanvasBindingPolicy bindings;
    private final List<CanvasGraphMutationPolicy> policies;
    private final CanvasPinnedOutputs pins;
    private final CanvasJson json;
    private final CanvasMcpConfigMedia mcpMedia;

    public CanvasDocument write(ProjectForkCommand request, CanvasProjectForkSources.Frozen frozen, long project,
                                CanvasProjectForkValues values, CanvasActor actor) {
        var original = frozen.manifest().graph();
        var target = commands.create(new CanvasCreateCommand(request.canvasId(), request.title(), project), actor);
        var copied = CanvasGraphCopies.copy(original, target.id());
        var nodes = copied.nodes().stream().map(node -> {
            var reference = node.resource();
            if (reference != null) {
                var source = reference;
                long version = frozen.objects().stream()
                        .filter(entry -> entry.type() == source.type() && entry.id() == source.id())
                        .findFirst().orElseThrow().data().path("version").longValue() + 1;
                reference = new CanvasResourceReference(reference.type(), values.id(reference.type(), reference.id()),
                        version);
            }
            var config = (ObjectNode) values.media(mcpMedia.rewrite(node, values::mediaReference), null);
            if (node.kind().storedMedia() && node.config().hasNonNull("resourceId")) {
                config.put("resourceId", values.mediaId(node.config().path("resourceId").asText()));
            }
            return new CanvasNode(node.id(), node.kind(), node.schemaVersion(), node.title(), node.layout(), config,
                    reference, 0, 0);
        }).toList();
        var graph = pins.copy(request.sourceCanvasId(), original, target.id(),
                new CanvasGraph(copied.schemaVersion(), project, nodes, copied.edges()), actor,
                value -> values.value(frozen.manifest().fixedOutputs().values().stream()
                        .filter(fixed -> fixed.id().equals(value.id())).findFirst().orElseThrow()));
        validator.validate(graph);
        bindings.validate(target, graph.nodes(), actor);
        var empty = new CanvasGraph(graph.schemaVersion(), project, List.of(), List.of());
        var mutation = new CanvasGraphMutation(graph, CanvasEditEngine.difference(empty, graph),
                CanvasEditEngine.difference(graph, empty));
        policies.forEach(policy -> policy.validate(target, empty, mutation, actor));
        graphs.save(target.id(), mutation);
        var receipt = new CanvasCommandReceipt(UUID.randomUUID().toString(), request.requestId(), target.id(),
                target.revision() + 1, target.nextEventSequence(), mutation.changes());
        journal.append(actor.userId(), "user", null, null, json.hash(request),
                Map.of("sourceCanvasId", request.sourceCanvasId(), "sourceRevision", request.expectedRevision(),
                        "projectId", project),
                receipt, mutation.previousValues(), "canvas.project.forked");
        documents.advance(target);
        return documents.require(target.id(), false);
    }
}
