package com.stonewu.fusion.service.canvas.template;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository;
import com.stonewu.fusion.service.canvas.*;
import com.stonewu.fusion.service.canvas.collaboration.CanvasGraphCopies;
import com.stonewu.fusion.service.canvas.collaboration.CanvasValueSnapshots;
import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import com.stonewu.fusion.service.canvas.command.CanvasCreateCommand;
import com.stonewu.fusion.service.canvas.command.CanvasEditEngine;
import com.stonewu.fusion.service.canvas.command.CanvasGraphMutation;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.resource.ProjectContentCopies.Copied;
import com.stonewu.fusion.service.resource.model.TemplateImportCommand;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.util.*;

@Component
@RequiredArgsConstructor
public class CanvasTemplateWriter {
    private final CanvasCommandService commands;
    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasJournalRepository journal;
    private final CanvasGraphValidator validator;
    private final CanvasBindingPolicy bindings;
    private final List<CanvasGraphMutationPolicy> policies;
    private final CanvasValueSnapshots snapshots;
    private final CanvasMcpConfigMedia mcp;
    private final CanvasJson json;
    private final CanvasRunSources sources;

    public CanvasDocument write(TemplateImportCommand request, CanvasTemplateBindingResolver.Resolved resolved,
                                Copied contents, CanvasActor actor) {
        var target = commands.create(new CanvasCreateCommand(request.canvasId(), request.title(), contents.projectId()),
                actor);
        var copied = CanvasGraphCopies.copy(resolved.graph(), target.id());
        var mapper = contents.remap();
        var nodes = copied.nodes().stream().map(node -> {
            var config = (ObjectNode) mapper.media(mcp.rewrite(node, mapper::mediaReference), null);
            if (node.kind().storedMedia() && node.config().hasNonNull("resourceId")) {
                config.put("resourceId", mapper.mediaId(node.config().path("resourceId").asText()));
            }
            var ref = node.resource();
            if (ref != null) {
                long id = mapper.id(ref.type(), ref.id());
                var type = ref.type();
                long version = contents.changes().stream()
                        .filter(change -> change.resourceType() == type && change.resourceId() == id).findFirst()
                        .orElseThrow().version();
                ref = new CanvasResourceReference(type, id, version);
            }
            return new CanvasNode(node.id(), node.kind(), node.schemaVersion(), node.title(), node.layout(), config,
                    ref, 0, 0);
        }).toList();
        var edges = new ArrayList<CanvasEdge>();
        String templateHash = json.hash(resolved.template());
        long fixedBytes = 0;
        for (int index = 0; index < copied.edges().size(); index++) {
            var edge = copied.edges().get(index);
            var original = resolved.template().fixedOutputs().get(resolved.graph().edges().get(index).id());
            if (original == null) {
                edges.add(edge);
                continue;
            }
            var value = mapper.value(original);
            if (original.type() == CanvasValueType.TEXT && original.mediaId() != null) {
                var fileNode = new CanvasNode(edge.sourceNodeId(), CanvasNodeKind.TEXT_FILE, 1, "模板固定文本", null,
                        json.object(Map.of("resourceId", value.mediaId())), null, 0, 0);
                var actual = sources.freeze(target, fileNode, actor, Set.of()).at("media").getFirst();
                value = new Value(value.id(), value.type(), actual.text(), value.mediaId(), value.mediaVersion(), null,
                        value.data(), null);
            }
            String valueId = UUID.nameUUIDFromBytes(
                    (target.id() + ":template-value:" + original.id()).getBytes(StandardCharsets.UTF_8)).toString();
            value = new Value(valueId, value.type(), value.text(), value.mediaId(), value.mediaVersion(),
                    value.resource(), value.data(), null);
            fixedBytes += json.write(value).getBytes(StandardCharsets.UTF_8).length;
            if (fixedBytes > CanvasTemplateData.MAX_BYTES) {
                throw new BusinessException(400, "导入后的固定内容超过 16 MiB，请缩小模板范围");
            }
            var output = snapshots.imported(templateHash, original.id(), target.id(), edge.sourceNodeId(),
                    edge.sourcePort(), value, actor);
            edges.add(
                    new CanvasEdge(edge.id(), edge.kind(), edge.sourceNodeId(), edge.sourcePort(), edge.targetNodeId(),
                            edge.targetPort(), edge.inputIndex(), output, 0));
        }
        var graph = new CanvasGraph(1, contents.projectId(), nodes, List.copyOf(edges));
        validator.validate(graph);
        bindings.validate(target, graph.nodes(), actor);
        var empty = new CanvasGraph(1, contents.projectId(), List.of(), List.of());
        var mutation = new CanvasGraphMutation(graph, CanvasEditEngine.difference(empty, graph),
                CanvasEditEngine.difference(graph, empty));
        policies.forEach(policy -> policy.validate(target, empty, mutation, actor));
        graphs.save(target.id(), mutation);
        var receipt = new CanvasCommandReceipt(UUID.randomUUID().toString(), request.requestId(), target.id(),
                target.revision() + 1, target.nextEventSequence(), mutation.changes());
        journal.append(actor.userId(), "user", null, null, json.hash(request),
                Map.of("templateHash", templateHash, "title", resolved.template().title()),
                receipt, mutation.previousValues(), "canvas.template.imported");
        documents.advance(target);
        return documents.require(target.id(), false);
    }
}
