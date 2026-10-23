package com.stonewu.fusion.service.canvas.collaboration;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository;
import com.stonewu.fusion.service.canvas.*;
import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import com.stonewu.fusion.service.canvas.command.CanvasCreateCommand;
import com.stonewu.fusion.service.canvas.command.CanvasEditEngine;
import com.stonewu.fusion.service.canvas.command.CanvasGraphMutation;
import com.stonewu.fusion.service.canvas.model.*;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * 创建新的图与审计来源，业务对象保持引用语义；独立业务分支由项目复制命令处理。
 */
@Service
@RequiredArgsConstructor
public class CanvasCheckpointForkService {
    private final CanvasWorkspaceService workspace;
    private final CanvasCommandService commands;
    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasJournalRepository journal;
    private final CanvasAccessPolicy access;
    private final CanvasBindingPolicy bindings;
    private final CanvasGraphValidator validator;
    private final List<CanvasGraphMutationPolicy> policies;
    private final CanvasJson json;
    private final CanvasPinnedOutputs pinned;

    public record Fork(@NotBlank @Pattern(regexp = "[A-Za-z0-9:._-]{1,96}") String requestId,
                       @NotBlank @Pattern(regexp = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}") String canvasId,
                       @NotBlank @Size(max = 200) String title) {
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasGraph", key = "#request.canvasId() + ':0'")
    public CanvasDocument fork(String source, String checkpoint, Fork request, CanvasActor actor) {
        var saved = workspace.checkpoint(source, checkpoint, actor);
        if (source.equals(request.canvasId())) {
            throw invalid("副本必须使用新的画布编号");
        }
        String hash = json.hash(Map.of("source", source, "checkpoint", checkpoint, "request", request));
        var existing = documents.find(request.canvasId(), true);
        if (existing.isPresent()) {
            access.requireView(existing.get(), actor);
            var replay = journal.find(request.canvasId(), actor.userId(), request.requestId());
            if (replay.isEmpty() || !replay.get().requestHash().equals(hash)) {
                throw invalid("目标画布编号已用于其他操作");
            }
            return existing.get();
        }
        var target = commands.create(
                new CanvasCreateCommand(request.canvasId(), request.title(), saved.snapshot().graph().projectId()),
                actor);
        var raced = journal.find(target.id(), actor.userId(), request.requestId());
        if (raced.isPresent()) {
            if (!raced.get().requestHash().equals(hash)) {
                throw invalid("目标画布已用于其他版本副本");
            }
            return target;
        }
        if (target.revision() != 1) {
            throw invalid("目标画布已有其他修改，不能覆盖为版本副本");
        }
        var graph = pinned.copy(source, saved.snapshot().graph(), target.id(),
                CanvasGraphCopies.copy(saved.snapshot().graph(), target.id()), actor, value -> value);
        validator.validate(graph);
        bindings.validate(target, graph.nodes(), actor);
        var empty = new CanvasGraph(graph.schemaVersion(), graph.projectId(), List.of(), List.of());
        var mutation = new CanvasGraphMutation(graph, CanvasEditEngine.difference(empty, graph),
                CanvasEditEngine.difference(graph, empty));
        policies.forEach(policy -> policy.validate(target, empty, mutation, actor));
        graphs.save(target.id(), mutation);
        var receipt = new CanvasCommandReceipt(UUID.randomUUID().toString(), request.requestId(), target.id(),
                target.revision() + 1,
                target.nextEventSequence(), mutation.changes());
        journal.append(actor.userId(), "user", null, null, hash,
                Map.of("sourceCanvasId", source, "checkpointId", checkpoint, "request", request),
                receipt, mutation.previousValues(), "canvas.checkpoint.forked");
        documents.advance(target);
        return documents.require(target.id(), false);
    }

    private CanvasException invalid(String message) {
        return new CanvasException(409, CanvasErrorCode.REQUEST_CONFLICT, message, Map.of());
    }
}
