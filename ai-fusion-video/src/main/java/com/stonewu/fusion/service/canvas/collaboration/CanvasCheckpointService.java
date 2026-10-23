package com.stonewu.fusion.service.canvas.collaboration;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository;
import com.stonewu.fusion.service.canvas.*;
import com.stonewu.fusion.service.canvas.command.CanvasCheckpointEngine;
import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import com.stonewu.fusion.service.canvas.command.CanvasGraphMutation;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;

@Service
@RequiredArgsConstructor
public class CanvasCheckpointService {
    private final CanvasWorkspaceService workspace;
    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasJournalRepository journal;
    private final CanvasAccessPolicy access;
    private final CanvasBindingPolicy bindings;
    private final CanvasCheckpointEngine engine;
    private final ResourceQueryService resources;
    private final List<CanvasGraphMutationPolicy> policies;
    private final CanvasJson json;

    public record NodeChange(String id, String title, String change, boolean content, boolean layout) {
    }

    public record ResourceChange(String type, long id, int savedVersion, int currentVersion) {
    }

    public record Plan(String checkpointId, String title, long expectedRevision, String digest, List<NodeChange> nodes,
                       int upsertedEdges, int removedEdges, List<ResourceChange> resources, boolean hasChanges) {
    }

    public record Restore(@NotBlank @Pattern(regexp = "[A-Za-z0-9:._-]{1,128}") String requestId,
                          @NotNull Long expectedRevision, @NotBlank String digest) {
    }

    private record Prepared(Plan plan, CanvasGraphMutation mutation) {
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Plan preview(String canvas, String checkpoint, CanvasActor actor) {
        var document = documents.require(canvas, false);
        access.requireEdit(document, actor);
        return prepare(document, checkpoint, actor).plan();
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasGraph", key = "#canvas + ':' + (#result.revision() - 1)")
    public CanvasCommandReceipt restore(String canvas, String checkpoint, Restore command, CanvasActor actor) {
        var document = documents.require(canvas, true);
        access.requireEdit(document, actor);
        String hash = json.hash(Map.of("checkpointId", checkpoint, "command", command));
        var replay = journal.find(canvas, actor.userId(), command.requestId());
        if (replay.isPresent()) {
            if (!replay.get().requestHash().equals(hash)) {
                throw conflict("此请求编号已用于其他恢复操作");
            }
            return replay.get().receipt();
        }
        if (!Objects.equals(command.expectedRevision(), document.revision())) {
            throw conflict("画布已变化，请重新比较命名版本");
        }
        var prepared = prepare(document, checkpoint, actor);
        if (!prepared.plan().digest().equals(command.digest())) {
            throw conflict("画布或引用内容在比较后发生变化，请重新确认");
        }
        if (!prepared.plan().hasChanges()) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "当前画布与此版本一致，无需恢复");
        }
        graphs.restore(canvas, prepared.mutation());
        var receipt = new CanvasCommandReceipt(UUID.randomUUID().toString(), command.requestId(), canvas,
                document.revision() + 1,
                document.nextEventSequence(), prepared.mutation().changes());
        journal.append(actor.userId(), "user", null, null, hash, Map.of("checkpointId", checkpoint, "command", command),
                receipt,
                prepared.mutation().previousValues(), "canvas.checkpoint.restored");
        documents.advance(document);
        return receipt;
    }

    private Prepared prepare(CanvasDocument document, String checkpoint, CanvasActor actor) {
        var saved = workspace.checkpoint(document.id(), checkpoint, actor);
        var current = graphs.load(document);
        var references = new ArrayList<ResourceChange>();
        for (var value : saved.snapshot().resources()) {
            var header = value.header();
            var latest = resources.get(header.resourceType(), header.resourceId(), actor).header();
            if (latest.version() != header.version()) {
                references.add(
                        new ResourceChange(header.resourceType().code(), header.resourceId(), header.version(),
                                latest.version()));
            }
        }
        var targetNodes = saved.snapshot().graph().nodes().stream().map(node -> {
            if (node.resource() == null) {
                return node;
            }
            var latest = resources.get(node.resource().type(), node.resource().id(), actor).header();
            return new CanvasNode(node.id(), node.kind(), node.schemaVersion(), node.title(), node.layout(),
                    node.config(),
                    new CanvasResourceReference(latest.resourceType(), latest.resourceId(), (long) latest.version()),
                    node.configVersion(), node.layoutVersion());
        }).toList();
        var desired = new CanvasGraph(saved.snapshot().graph().schemaVersion(), saved.snapshot().graph().projectId(),
                targetNodes, saved.snapshot().graph().edges());
        var mutation = engine.restore(current, desired,
                graphs.deletedNodes(document.id(), desired.nodes().stream().map(CanvasNode::id).toList()),
                graphs.deletedEdges(document.id(), desired.edges().stream().map(edge -> edge.id()).toList()));
        bindings.validate(document, desired.nodes(), actor);
        policies.forEach(policy -> policy.validate(document, current, mutation, actor));
        List<NodeChange> changed = new ArrayList<>();
        for (var node : mutation.changes().upsertedNodes()) {
            var previous = current.nodes().stream().filter(old -> old.id().equals(node.id())).findFirst();
            changed.add(new NodeChange(node.id(), node.title(), previous.isPresent() ? "MODIFIED" : "RESTORED",
                    previous.isEmpty() || previous.get().configVersion() != node.configVersion(),
                    previous.isEmpty() || previous.get().layoutVersion() != node.layoutVersion()));
        }
        for (String id : mutation.changes().deletedNodeIds()) {
            var previous = current.nodes().stream().filter(node -> node.id().equals(id)).findFirst().orElseThrow();
            changed.add(new NodeChange(id, previous.title(), "REMOVED", true, true));
        }
        String digest = json.hash(
                Map.of("checkpoint", checkpoint, "actor", actor, "revision", document.revision(), "mutation", mutation,
                        "resources", references));
        var plan = new Plan(checkpoint, saved.checkpoint().title(), document.revision(), digest, changed,
                mutation.changes().upsertedEdges().size(),
                mutation.changes().deletedEdgeIds().size(), references,
                !changed.isEmpty() || !mutation.changes().upsertedEdges().isEmpty() || !mutation.changes()
                        .deletedEdgeIds().isEmpty());
        return new Prepared(plan, mutation);
    }

    private CanvasException conflict(String message) {
        return new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT, message, Map.of());
    }
}
