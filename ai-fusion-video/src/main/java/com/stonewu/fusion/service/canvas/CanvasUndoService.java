package com.stonewu.fusion.service.canvas;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository;
import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import com.stonewu.fusion.service.canvas.command.CanvasUndoCommand;
import com.stonewu.fusion.service.canvas.command.CanvasUndoEngine;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class CanvasUndoService {
    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasJournalRepository journal;
    private final CanvasUndoEngine undo;
    private final CanvasAccessPolicy access;
    private final CanvasBindingPolicy bindings;
    private final CanvasRequestValidator requests;
    private final CanvasJson json;
    private final List<CanvasGraphMutationPolicy> policies;
    private final ResourceAgentFence agents;

    @Transactional
    @CacheEvict(value = "canvasGraph", key = "#canvasId + ':' + (#result.revision() - 1)")
    public CanvasCommandReceipt undo(String canvasId, CanvasUndoCommand command, CanvasActor actor) {
        return undo(canvasId, command, ResourceCommandContext.user(actor, canvasId));
    }

    @Transactional
    @CacheEvict(value = "canvasGraph", key = "#canvasId + ':' + (#result.revision() - 1)")
    public CanvasCommandReceipt undo(String canvasId, CanvasUndoCommand command, ResourceCommandContext context) {
        CanvasActor actor = context.actor();
        if (!Objects.equals(canvasId, context.canvasId())) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "命令来源与画布不一致");
        }
        requests.id(canvasId);
        requests.id(command.commandId());
        requests.validate(command);
        var document = documents.require(canvasId, true);
        access.requireEdit(document, actor);
        agents.requireActive(context, document.projectId());
        var origin = json.object(context);
        origin.remove(List.of("ownerInstanceId", "ownerEpoch"));
        String hash = "user".equals(context.origin()) ? json.hash(command) : json.hash(
                Map.of("command", command, "context", origin));
        var replay = journal.find(canvasId, actor.userId(), command.requestId());
        if (replay.isPresent()) {
            if (!replay.get().requestHash().equals(hash)) {
                throw new CanvasException(409, CanvasErrorCode.REQUEST_CONFLICT,
                        "同一请求 ID 不能提交不同的内容", Map.of());
            }
            return replay.get().receipt();
        }
        var source = journal.undoSource(canvasId, command.commandId());
        var graph = graphs.load(document);
        var mutation = undo.apply(graph, source.receipt().changes(), source.before(),
                graphs.deletedNodes(canvasId, source.receipt().changes().deletedNodeIds()),
                graphs.deletedEdges(canvasId, source.receipt().changes().deletedEdgeIds()));
        bindings.validate(document, mutation.changes().upsertedNodes().stream()
                .filter(node -> graph.nodes().stream().noneMatch(
                        old -> old.id().equals(node.id()) && Objects.equals(old.resource(), node.resource())))
                .toList(), actor);
        policies.forEach(policy -> policy.validate(document, graph, mutation, actor));
        graphs.restore(canvasId, mutation);
        var receipt = new CanvasCommandReceipt(UUID.randomUUID().toString(), command.requestId(), canvasId,
                document.revision() + 1, document.nextEventSequence(), mutation.changes());
        journal.append(actor.userId(), context.origin(), context.agentRunId(), context.toolCallId(), hash,
                command, receipt, mutation.previousValues(), "canvas.graph.undone");
        documents.advance(document);
        return receipt;
    }
}
