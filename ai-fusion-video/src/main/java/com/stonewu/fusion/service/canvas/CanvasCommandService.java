package com.stonewu.fusion.service.canvas;

import com.stonewu.fusion.repository.canvas.CanvasAccessRepository.ProjectScope;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository.StoredCommand;
import com.stonewu.fusion.service.canvas.command.*;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import java.util.*;
import java.util.function.BiFunction;

@Service
public class CanvasCommandService {
    private static final CanvasGraphChangeSet EMPTY = new CanvasGraphChangeSet(List.of(), List.of(), List.of(),
            List.of());
    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasJournalRepository journal;
    private final CanvasEditEngine edits;
    private final CanvasAccessPolicy access;
    private final CanvasBindingPolicy bindings;
    private final CanvasRequestValidator requests;
    private final CanvasJson json;
    private final TransactionTemplate writes;
    private final List<CanvasGraphMutationPolicy> policies;
    private final ResourceAgentFence agents;

    public CanvasCommandService(CanvasDocumentRepository documents, CanvasGraphRepository graphs,
                                CanvasJournalRepository journal, CanvasEditEngine edits, CanvasAccessPolicy access,
                                CanvasRequestValidator requests, CanvasJson json, CanvasBindingPolicy bindings,
                                PlatformTransactionManager manager, List<CanvasGraphMutationPolicy> policies,
                                ResourceAgentFence agents) {
        this.documents = documents;
        this.graphs = graphs;
        this.journal = journal;
        this.edits = edits;
        this.access = access;
        this.bindings = bindings;
        this.requests = requests;
        this.json = json;
        this.writes = new TransactionTemplate(manager);
        this.policies = List.copyOf(policies);
        this.agents = agents;
    }

    @CacheEvict(value = "canvasGraph", key = "#result.id() + ':' + (#result.revision() - 1)")
    public CanvasDocument create(CanvasCreateCommand command, CanvasActor actor) {
        requests.validate(command);
        String hash = json.hash(command);
        return Objects.requireNonNull(writes.execute(ignored -> {
            access.requireActorScope(actor);
            if (command.projectId() != null) {
                access.requireProjectEditor(command.projectId(), actor);
            }
            ProjectScope scope = command.projectId() == null ? null : access.requireProject(command.projectId(), actor);
            documents.admit(command.id(), command.title(), command.projectId(), scope == null ? 1 : scope.ownerType(),
                    scope == null ? actor.userId() : scope.ownerId(), actor.userId());
            CanvasDocument document = documents.require(command.id(), true);
            access.requireView(document, actor);
            String requestId = "create:" + command.id();
            Optional<StoredCommand> previous = journal.find(command.id(), actor.userId(), requestId);
            if (previous.isPresent()) {
                requireMatching(previous.get(), hash);
                return document;
            }
            if (document.createdBy() != actor.userId() || document.revision() != 0) {
                throw requestConflict();
            }
            CanvasCommandReceipt receipt = receipt(document, requestId, EMPTY);
            journal.append(actor.userId(), "user", null, null, hash, command, receipt, EMPTY, "canvas.created");
            documents.advance(document);
            return documents.require(document.id(), false);
        }));
    }

    @CacheEvict(value = "canvasGraph", key = "#canvasId + ':' + (#result.revision() - 1)")
    public CanvasCommandReceipt edit(String canvasId, CanvasEditCommand command, CanvasActor actor) {
        return edit(canvasId, command, ResourceCommandContext.user(actor, canvasId));
    }

    @CacheEvict(value = "canvasGraph", key = "#canvasId + ':' + (#result.revision() - 1)")
    public CanvasCommandReceipt edit(String canvasId, CanvasEditCommand command, ResourceCommandContext context) {
        requests.validate(command);
        var origin = json.object(context);
        origin.remove(List.of("ownerInstanceId", "ownerEpoch"));
        String hash = "user".equals(context.origin()) ? json.hash(command) : json.hash(
                Map.of("command", command, "context", origin));
        return commit(canvasId, command.requestId(), context, hash, (document, graph) -> command);
    }

    /**
     * 衍生操作在持有画布锁后计算一次；重放先匹配原始意图，不重新基于新布局计算。
     */
    @CacheEvict(value = "canvasGraph", key = "#canvasId + ':' + (#result.revision() - 1)")
    public CanvasCommandReceipt derive(String canvasId, String requestId, Object intent, ResourceCommandContext context,
                                       BiFunction<CanvasDocument, CanvasGraph, CanvasEditCommand> prepare) {
        if (!"agent".equals(context.origin())) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "衍生命令需要可信智能体上下文");
        }
        var origin = json.object(context);
        origin.remove(List.of("ownerInstanceId", "ownerEpoch"));
        return commit(canvasId, requestId, context, json.hash(Map.of("derivedIntent", intent, "context", origin)),
                prepare);
    }

    private CanvasCommandReceipt commit(String canvasId, String requestId, ResourceCommandContext context, String hash,
                                        BiFunction<CanvasDocument, CanvasGraph, CanvasEditCommand> prepare) {
        CanvasActor actor = context.actor();
        if (!Objects.equals(context.canvasId(), canvasId)) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "命令来源与画布不一致");
        }
        requests.id(canvasId);
        if (requestId == null || !requestId.matches("[A-Za-z0-9._:-]{1,128}")) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "请求标识无效");
        }
        try {
            return Objects.requireNonNull(writes.execute(ignored -> {
                CanvasDocument document = documents.require(canvasId, true);
                access.requireEdit(document, actor);
                agents.requireActive(context, document.projectId());
                Optional<StoredCommand> previous = journal.find(canvasId, actor.userId(), requestId);
                if (previous.isPresent()) {
                    return requireMatching(previous.get(), hash);
                }
                CanvasGraph graph = graphs.load(document);
                CanvasEditCommand command = prepare.apply(document, graph);
                requests.validate(command);
                if (!requestId.equals(command.requestId())) {
                    throw new IllegalStateException("衍生命令标识不一致");
                }
                CanvasGraphMutation mutation = edits.apply(graph, command.operations());
                validateNewBindings(document, graph, mutation, actor);
                policies.forEach(policy -> policy.validate(document, graph, mutation, actor));
                graphs.save(canvasId, mutation);
                CanvasCommandReceipt receipt = receipt(document, command.requestId(), mutation.changes());
                journal.append(actor.userId(), context.origin(), context.agentRunId(), context.toolCallId(), hash,
                        command, receipt, mutation.previousValues(), "canvas.graph.changed");
                documents.advance(document);
                return receipt;
            }));
        } catch (DuplicateKeyException conflict) {
            // 事务已经回滚；禁止复用其他画布或已删除节点的 ID。
            throw new CanvasException(409, CanvasErrorCode.REQUEST_CONFLICT,
                    "节点或连线 ID 已被使用，请为新对象生成新的 ID", Map.of());
        }
    }

    private void validateNewBindings(CanvasDocument document, CanvasGraph before, CanvasGraphMutation mutation,
                                     CanvasActor actor) {
        Map<String, CanvasNode> old = new HashMap<>();
        before.nodes().forEach(node -> old.put(node.id(), node));
        bindings.validate(document, mutation.changes().upsertedNodes().stream()
                .filter(node -> !old.containsKey(node.id()) || !Objects.equals(old.get(node.id()).resource(),
                        node.resource()))
                .toList(), actor);
    }

    private CanvasCommandReceipt receipt(CanvasDocument document, String requestId, CanvasGraphChangeSet changes) {
        return new CanvasCommandReceipt(UUID.randomUUID().toString(), requestId, document.id(),
                document.revision() + 1, document.nextEventSequence(), changes);
    }

    private CanvasCommandReceipt requireMatching(StoredCommand command, String hash) {
        if (!command.requestHash().equals(hash)) {
            throw requestConflict();
        }
        return command.receipt();
    }

    private CanvasException requestConflict() {
        return new CanvasException(409, CanvasErrorCode.REQUEST_CONFLICT, "同一请求 ID 不能提交不同的内容", Map.of());
    }
}
