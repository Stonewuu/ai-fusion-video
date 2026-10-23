package com.stonewu.fusion.service.canvas;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository.HistoryEntry;
import com.stonewu.fusion.repository.canvas.CanvasWorkspaceRepository;
import com.stonewu.fusion.repository.canvas.CanvasWorkspaceRepository.Checkpoint;
import com.stonewu.fusion.repository.canvas.CanvasWorkspaceRepository.CheckpointContent;
import com.stonewu.fusion.repository.canvas.CanvasWorkspaceRepository.CheckpointDetail;
import com.stonewu.fusion.repository.canvas.CanvasWorkspaceRepository.Comment;
import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import com.stonewu.fusion.service.canvas.command.CanvasGraphChangeSet;
import com.stonewu.fusion.service.canvas.command.CanvasWorkspaceCommands.CheckpointCreate;
import com.stonewu.fusion.service.canvas.command.CanvasWorkspaceCommands.CommentCreate;
import com.stonewu.fusion.service.canvas.command.CanvasWorkspaceCommands.CommentPatch;
import com.stonewu.fusion.service.canvas.command.CanvasWorkspaceCommands.DocumentPatch;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceSnapshot;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;
import java.util.function.Consumer;

@Service
@RequiredArgsConstructor
public class CanvasWorkspaceService {
    private static final CanvasGraphChangeSet EMPTY = new CanvasGraphChangeSet(List.of(), List.of(), List.of(),
            List.of());
    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasWorkspaceRepository workspace;
    private final CanvasJournalRepository journal;
    private final CanvasAccessPolicy access;
    private final ResourceQueryService resources;
    private final CanvasRequestValidator requests;
    private final CanvasJson json;
    private final ResourceAgentFence agents;

    @Transactional
    @CacheEvict(value = "canvasGraph", key = "#canvasId + ':' + (#result.revision() - 1)")
    public CanvasCommandReceipt patch(String canvasId, DocumentPatch command, CanvasActor actor) {
        return mutate(canvasId, command.requestId(), command, actor, "canvas.document.changed", document -> {
            if (command.archived() == null) {
                access.requireEdit(document, actor);
            } else {
                access.requireManage(document, actor);
            }
        }, document -> {
            version(command.expectedRevision(), document.revision());
            if (command.title() == null && command.archived() == null) {
                throw invalid("必须指定名称或归档状态");
            }
            if (command.title() != null && command.title().isBlank()) {
                throw invalid("画布名称不能为空");
            }
            workspace.patchDocument(canvasId, command.title() == null ? document.title() : command.title(),
                    command.archived() == null ? document.archived() : command.archived());
        });
    }

    @Transactional
    @CacheEvict(value = "canvasGraph", key = "#canvasId + ':' + (#result.revision() - 1)")
    public CanvasCommandReceipt comment(String canvasId, CommentCreate command, CanvasActor actor) {
        return mutate(canvasId, command.requestId(), command, actor, "canvas.comment.created",
                document -> access.requireEdit(document, actor), document -> {
                    requests.id(command.id());
                    if (command.nodeId() != null) {
                        requests.id(command.nodeId());
                    }
                    if (command.parentId() != null) {
                        requests.id(command.parentId());
                        var parent = workspace.comment(canvasId, command.parentId());
                        if (parent.deleted() || parent.parentId() != null || !Objects.equals(parent.nodeId(),
                                command.nodeId())) {
                            throw invalid("只能回复同一锚点的有效主评论");
                        }
                    }
                    Long resourceVersion = null;
                    if (command.nodeId() != null) {
                        var node = graphs.load(document).nodes().stream()
                                .filter(item -> item.id().equals(command.nodeId())).findFirst()
                                .orElseThrow(() -> invalid("评论锚定的节点已删除"));
                        if (node.resource() != null) {
                            resourceVersion = (long) resources.get(node.resource().type(), node.resource().id(), actor)
                                    .header().version();
                        }
                    }
                    workspace.createComment(canvasId, command.id(), command.nodeId(), command.parentId(),
                            resourceVersion, command.content(), actor.userId());
                });
    }

    @Transactional
    @CacheEvict(value = "canvasGraph", key = "#canvasId + ':' + (#result.revision() - 1)")
    public CanvasCommandReceipt patchComment(String canvasId, String commentId, CommentPatch command,
                                             CanvasActor actor) {
        return mutate(canvasId, command.requestId(), Map.of("commentId", commentId, "command", command), actor,
                "canvas.comment.changed", document -> access.requireEdit(document, actor), document -> {
                    requests.validate(command);
                    requests.id(commentId);
                    var previous = workspace.comment(canvasId, commentId);
                    version(command.expectedVersion(), previous.version());
                    if (previous.deleted()) {
                        throw invalid("评论已删除");
                    }
                    if (command.content() == null && command.resolved() == null && command.deleted() == null) {
                        throw invalid("评论修改不能为空");
                    }
                    if ((command.content() != null || command.deleted() != null) && previous.createdBy() != actor.userId()) {
                        access.requireManage(document, actor);
                    }
                    if (command.content() != null && command.content().isBlank()) {
                        throw invalid("评论内容不能为空");
                    }
                    if (Boolean.FALSE.equals(command.deleted())) {
                        throw invalid("删除后的评论不可通过修改恢复");
                    }
                    workspace.patchComment(canvasId, previous,
                            command.content() == null ? previous.content() : command.content(),
                            command.resolved() == null ? previous.resolved() : command.resolved(),
                            Boolean.TRUE.equals(command.deleted()));
                });
    }

    @Transactional(isolation = Isolation.REPEATABLE_READ)
    @CacheEvict(value = "canvasGraph", key = "#canvasId + ':' + (#result.revision() - 1)")
    public CanvasCommandReceipt checkpoint(String canvasId, CheckpointCreate command, CanvasActor actor) {
        return checkpoint(canvasId, command, ResourceCommandContext.user(actor, canvasId));
    }

    @Transactional(isolation = Isolation.REPEATABLE_READ)
    @CacheEvict(value = "canvasGraph", key = "#canvasId + ':' + (#result.revision() - 1)")
    public CanvasCommandReceipt checkpoint(String canvasId, CheckpointCreate command, ResourceCommandContext context) {
        CanvasActor actor = context.actor();
        return mutate(canvasId, command.requestId(), command, context, "canvas.checkpoint.created",
                document -> access.requireEdit(document, actor), document -> {
                    requests.id(command.id());
                    version(command.expectedRevision(), document.revision());
                    var graph = graphs.load(document);
                    Map<String, ResourceSnapshot> contents = new LinkedHashMap<>();
                    for (var node : graph.nodes()) {
                        if (node.resource() == null) {
                            continue;
                        }
                        var ref = node.resource();
                        String key = ref.type().code() + ":" + ref.id();
                        contents.computeIfAbsent(key, ignored -> resources.get(ref.type(), ref.id(), actor));
                    }
                    workspace.createCheckpoint(canvasId, command.id(), command.title(), document.revision(),
                            document.nextEventSequence() - 1,
                            actor.userId(), new CheckpointContent(graph, List.copyOf(contents.values())));
                });
    }

    public List<HistoryEntry> history(String canvasId, CanvasActor actor, long beforeRevision, int limit) {
        requireRead(canvasId, actor);
        pagination(beforeRevision, limit);
        return journal.history(canvasId, beforeRevision, limit);
    }

    public List<Comment> comments(String canvasId, CanvasActor actor, int offset, int limit) {
        requireRead(canvasId, actor);
        pagination(offset, limit);
        return workspace.comments(canvasId, offset, limit);
    }

    public Comment comment(String canvasId, String commentId, CanvasActor actor) {
        requireRead(canvasId, actor);
        requests.id(commentId);
        return workspace.comment(canvasId, commentId);
    }

    public List<Checkpoint> checkpoints(String canvasId, CanvasActor actor, long beforeRevision, int limit) {
        requireRead(canvasId, actor);
        pagination(beforeRevision, limit);
        return workspace.checkpoints(canvasId, beforeRevision, limit);
    }

    public CheckpointDetail checkpoint(String canvasId, String id, CanvasActor actor) {
        requireRead(canvasId, actor);
        requests.id(id);
        var result = workspace.checkpoint(canvasId, id);
        // 历史正文不能凭过去的分享授权越过当前业务权限。
        for (var resource : result.snapshot().resources()) {
            resources.authorizeHistorical(resource.header().resourceType(), resource.header().resourceId(), actor);
        }
        return result;
    }

    private CanvasCommandReceipt mutate(String canvasId, String requestId, Object command, CanvasActor actor,
                                        String eventType,
                                        Consumer<CanvasDocument> authorize, Consumer<CanvasDocument> mutation) {
        return mutate(canvasId, requestId, command, ResourceCommandContext.user(actor, canvasId), eventType, authorize,
                mutation);
    }

    private CanvasCommandReceipt mutate(String canvasId, String requestId, Object command,
                                        ResourceCommandContext context, String eventType,
                                        Consumer<CanvasDocument> authorize, Consumer<CanvasDocument> mutation) {
        CanvasActor actor = context.actor();
        if (!Objects.equals(canvasId, context.canvasId())) {
            throw invalid("命令来源与画布不一致");
        }
        requests.id(canvasId);
        requests.validate(command);
        if (requestId == null || !requestId.matches("[A-Za-z0-9._:-]{1,128}")) {
            throw invalid("请求标识无效");
        }
        var document = documents.require(canvasId, true);
        authorize.accept(document);
        agents.requireActive(context, document.projectId());
        var origin = json.object(context);
        origin.remove(List.of("ownerInstanceId", "ownerEpoch"));
        String hash = "user".equals(context.origin()) ? json.hash(Map.of("type", eventType, "command", command))
                : json.hash(Map.of("type", eventType, "command", command, "context", origin));
        var replay = journal.find(canvasId, actor.userId(), requestId);
        if (replay.isPresent()) {
            if (!replay.get().requestHash().equals(hash)) {
                throw new CanvasException(409, CanvasErrorCode.REQUEST_CONFLICT, "请求标识已用于不同操作", Map.of());
            }
            return replay.get().receipt();
        }
        try {
            mutation.accept(document);
        } catch (DuplicateKeyException conflict) {
            throw new CanvasException(409, CanvasErrorCode.REQUEST_CONFLICT, "评论或命名版本 ID 已被使用", Map.of());
        }
        var receipt = new CanvasCommandReceipt(UUID.randomUUID().toString(), requestId, canvasId,
                document.revision() + 1, document.nextEventSequence(), EMPTY);
        journal.append(actor.userId(), context.origin(), context.agentRunId(), context.toolCallId(), hash, command,
                receipt, EMPTY, eventType);
        documents.advance(document);
        return receipt;
    }

    private void requireRead(String canvasId, CanvasActor actor) {
        requests.id(canvasId);
        access.requireView(documents.require(canvasId, false), actor);
    }

    private void pagination(long cursor, int limit) {
        if (cursor < 0 || limit < 1 || limit > 100) {
            throw invalid("分页参数无效，每页最多 100 项");
        }
    }

    private void version(Long expected, long actual) {
        if (expected == null || expected != actual) {
            throw new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT,
                    "内容已改变，请重新读取后操作", Map.of("actual", actual));
        }
    }

    private CanvasException invalid(String message) {
        return new CanvasException(CanvasErrorCode.INVALID_COMMAND, message);
    }
}
