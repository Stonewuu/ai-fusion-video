package com.stonewu.fusion.service.canvas.run;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.run.writeback.CanvasWritebackService;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

@Service
@RequiredArgsConstructor
public class CanvasRunService {
    private final CanvasRunRepository runs;
    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasAccessPolicy access;
    private final CanvasRunPlanner planner;
    private final CanvasRunSources sources;
    private final CanvasRunInputs inputs;
    private final CanvasRunEnvironment environments;
    private final CanvasRunStatistics statistics;
    private final ResourceAgentFence agents;
    private final CanvasJson json;

    @Transactional(isolation = Isolation.REPEATABLE_READ)
    @CacheEvict(value = "canvasRunPlan", key = "#canvasId")
    public PlanView plan(String canvasId, PlanRequest request, CanvasActor actor) {
        request.scope().require(actor);
        var document = documents.require(canvasId, false);
        access.requireExecute(document, actor);
        if (request.expectedRevision() == null || request.expectedRevision() != document.revision()) {
            throw CanvasRunException.conflict("画布已变化，请等待同步后重新预览");
        }
        var plan = planner.build(document, graphs.load(document), request, actor);
        if (json.write(plan).length() > 8 * 1024 * 1024) {
            throw CanvasRunException.invalid("运行快照超过 8 MiB，请缩小范围");
        }
        runs.storePlan(plan);
        return planner.view(plan);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasRun", allEntries = true)
    public View start(String canvasId, Start request, ResourceCommandContext context) {
        request.scope().require(context.actor());
        requestId(request.requestId());
        runs.lockActor(context.actor());
        var document = documents.require(canvasId, true);
        access.requireExecute(document, context.actor());
        agents.requireActive(context, document.projectId());
        if (!Objects.equals(context.canvasId(), canvasId)) {
            throw CanvasRunException.invalid("运行来源与画布不一致");
        }
        var receipt = runs.receipt(request.requestId(), context.actor());
        if (receipt != null) {
            if (!receipt.hash().equals(json.hash(request)) || !runs.run(receipt.runId(), false).canvasId()
                    .equals(canvasId)) {
                throw CanvasRunException.conflict("请求标识已用于不同的运行计划");
            }
            return view(receipt.runId(), context.actor());
        }
        if (runs.withdrawn(request.requestId(), context.actor())) {
            throw new CanvasRunException(410, "RUN_REQUEST_WITHDRAWN", "此运行请求已撤回，不会执行迟到的提交");
        }
        var stored = runs.plan(request.planId());
        var plan = stored.value();
        if (!plan.actor().equals(context.actor()) || !plan.canvasId().equals(canvasId)) {
            throw new CanvasRunException(403, "PLAN_SCOPE_MISMATCH", "运行计划不属于当前账号、团队或画布");
        }
        if (!stored.hash().equals(request.planHash()) || !plan.expiresAt().isAfter(Instant.now())) {
            throw CanvasRunException.conflict("计划已过期或内容不一致，请重新预览");
        }
        if (!plan.issues().isEmpty()) {
            throw CanvasRunException.invalid("请先解决运行计划中的问题");
        }
        if (document.revision() != plan.documentRevision()) {
            throw CanvasRunException.conflict("画布在预览后发生变化，请重新确认范围");
        }
        if (runs.active(context.actor()) >= 8) {
            throw CanvasRunException.conflict("同时最多保留 8 个未结束的画布运行，请先处理现有任务");
        }
        Map<String, String> reuse = plan.nodes().stream().filter(node -> node.reusedNodeRunId() != null)
                .collect(Collectors.toMap(node -> node.node().id(), NodePlan::reusedNodeRunId));
        var checked = planner.build(document, graphs.load(document),
                new PlanRequest(plan.documentRevision(), plan.mode(),
                        plan.requested(), reuse, plan.budget(), request.scope()), context.actor());
        if (!checked.issues().isEmpty() || !fingerprints(plan).equals(fingerprints(checked))) {
            throw CanvasRunException.conflict("输入资源、模型能力或运行范围在预览后发生变化，请重新确认");
        }
        if (runs.hasActiveNodes(canvasId, checked.nodes().stream().filter(node -> !node.action().equals("REUSE"))
                .map(node -> node.node().id()).toList())) {
            throw CanvasRunException.conflict("本次范围内有节点正在运行，请等待完成后再运行");
        }
        String id = runs.insert(plan, request, context);
        return view(id, context.actor());
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View view(String id, CanvasActor actor) {
        var run = runs.run(id, false);
        var document = documents.require(run.canvasId(), false);
        var permissions = access.requireView(document, actor);
        CanvasGraph graph = graphs.load(document);
        var current = graph.nodes().stream().collect(Collectors.toMap(CanvasNode::id, Function.identity()));
        var states = derived(run, graph, current, actor);
        var executionNodes = runs.nodes(id);
        Set<String> unavailable = new HashSet<>();
        var nodes = executionNodes.stream().map(node -> {
            if (!sources.visible(node, actor)) {
                unavailable.add(node.id());
                return new NodeView(node.id(), node.nodeId(), "内容不可访问", node.plan().node().kind().code(),
                        node.plan().action(), node.status(),
                        "部分输入或输出已删除或失去访问权限，内容已隐藏；仍可停止本次运行。",
                        node.revision(), null, Ports.empty(), null, null, "SOURCE_UNAVAILABLE", node.parentNodeRunId(),
                        node.itemIndex(), node.attempt(), null, true);
            }
            return new NodeView(node.id(), node.nodeId(), node.plan().node().title(), node.plan().node().kind().code(),
                    node.plan().action(), node.status(), node.error(), node.revision(), node.inputs(), node.outputs(),
                    node.externalKind(), node.externalId(), states.get(node.nodeId()), node.parentNodeRunId(),
                    node.itemIndex(), node.attempt(), CanvasWritebackService.publicView(node), false);
        }).toList();
        return new View(run.id(), run.canvasId(), run.plan().documentRevision(), run.plan().mode(), run.status(),
                run.cancelRequested(), run.revision(), run.error(), run.plan().cost(), run.createdAt(), run.updatedAt(),
                nodes,
                run.actor().equals(actor) && permissions.canExecute(),
                statistics.read(run, executionNodes, unavailable));
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public List<Summary> list(String canvasId, int page, CanvasActor actor) {
        if (page < 0 || page > 10000) {
            throw CanvasRunException.invalid("运行历史分页无效");
        }
        access.requireView(documents.require(canvasId, false), actor);
        return runs.history(canvasId, page).stream().map(run -> new Summary(run.id(), run.status(), run.plan().mode(),
                run.revision(), run.createdAt(), run.plan().nodes().size())).toList();
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public List<NodeActivity> nodeActivities(String canvasId, CanvasActor actor) {
        var document = documents.require(canvasId, false);
        var permissions = access.requireView(document, actor);
        CanvasGraph graph = graphs.load(document);
        var current = graph.nodes().stream().collect(Collectors.toMap(CanvasNode::id, Function.identity()));
        Map<String, Run> parents = new HashMap<>();
        Map<String, Map<String, String>> states = new HashMap<>();
        Map<String, List<NodeRun>> children = new HashMap<>();
        return runs.latestNodes(canvasId).stream().filter(node -> current.containsKey(node.nodeId())).map(node -> {
            Run run = parents.computeIfAbsent(node.runId(), id -> runs.run(id, false));
            var derivedStates = states.computeIfAbsent(run.id(), id -> derived(run, graph, current, actor));
            var nested = "BATCH".equals(node.externalKind())
                    ? children.computeIfAbsent(run.id(), runs::nodes).stream()
                    .filter(child -> node.id().equals(child.parentNodeRunId()))
                    .map(child -> publicNode(child, derivedStates, actor)).toList()
                    : List.<NodeView>of();
            return new NodeActivity(run.id(), run.status(), run.actor().equals(actor) && permissions.canExecute(),
                    publicNode(node, derivedStates, actor), nested);
        }).toList();
    }

    private NodeView publicNode(NodeRun node, Map<String, String> states, CanvasActor actor) {
        if (!sources.visible(node, actor)) {
            return new NodeView(node.id(), node.nodeId(), "内容不可访问", node.plan().node().kind().code(),
                    node.plan().action(), node.status(), "部分输入或输出已删除或失去访问权限，内容已隐藏。",
                    node.revision(), null, Ports.empty(), null, null, "SOURCE_UNAVAILABLE", node.parentNodeRunId(),
                    node.itemIndex(), node.attempt(), null, true);
        }
        return new NodeView(node.id(), node.nodeId(), node.plan().node().title(), node.plan().node().kind().code(),
                node.plan().action(), node.status(), node.error(), node.revision(), node.inputs(), node.outputs(),
                node.externalKind(), node.externalId(), states.get(node.nodeId()), node.parentNodeRunId(),
                node.itemIndex(),
                node.attempt(), CanvasWritebackService.publicView(node), false);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasRun", key = "#id")
    public View cancel(String id, GenerationClientScope scope, CanvasActor actor) {
        return cancel(id, scope, actor, null);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasRun", key = "#id")
    public View cancel(String id, GenerationClientScope scope, CanvasActor actor, ResourceCommandContext context) {
        scope.require(actor);
        var run = control(id, actor);
        requireControlContext(run, context);
        if (!run.status().terminal() && !run.cancelRequested()) {
            runs.requestCancel(id);
            runs.changeRun(id, State.CANCEL_REQUESTED, null, "CANVAS_RUN_CANCELLED",
                    Map.of("actorUserId", actor.userId()));
        }
        return view(id, actor);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasRun", allEntries = true)
    public View withdraw(String canvasId, String requestId, GenerationClientScope scope, CanvasActor actor) {
        scope.require(actor);
        requestId(requestId);
        runs.lockActor(actor);
        access.requireView(documents.require(canvasId, true), actor);
        var receipt = runs.receipt(requestId, actor);
        if (receipt != null && !runs.run(receipt.runId(), false).canvasId().equals(canvasId)) {
            throw CanvasRunException.conflict("原请求属于另一张画布");
        }
        runs.withdraw(canvasId, requestId, actor);
        return receipt == null ? null : cancel(receipt.runId(), scope, actor);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasRun", key = "#id")
    public View select(String id, String nodeRunId, Selection request, CanvasActor actor) {
        return select(id, nodeRunId, request, actor, null);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasRun", key = "#id")
    public View select(String id, String nodeRunId, Selection request, CanvasActor actor,
                       ResourceCommandContext context) {
        request.scope().require(actor);
        var run = control(id, actor);
        requireControlContext(run, context);
        if (run.cancelRequested() || !run.deadlineAt().isAfter(Instant.now())) {
            throw CanvasRunException.fenced();
        }
        var node = runs.node(nodeRunId, true);
        if (!node.runId().equals(id)) {
            throw CanvasRunException.invalid("候选节点不属于此运行");
        }
        // 重复选择同一候选是幂等的，不能用重试覆盖已确认的另一个候选。
        if (node.status() == State.SUCCEEDED && request.outputId()
                .equals(node.checkpoint().path("selectedInputId").asText())) {
            return view(id, actor);
        }
        if (node.status() != State.WAITING_SELECTION || node.revision() != request.expectedRevision()) {
            throw CanvasRunException.conflict("候选等待状态已变化，请刷新后重试");
        }
        var selected = node.inputs().at("candidates").stream().filter(value -> value.id().equals(request.outputId()))
                .findFirst()
                .orElseThrow(() -> CanvasRunException.invalid("所选结果不在本次候选范围内"));
        sources.authorize(selected, actor, true);
        var checkpoint = node.checkpoint().deepCopy().put("selectedInputId", selected.id());
        var output = inputs.publish(node, new Ports(Map.of("selected", List.of(selected))));
        runs.updateNode(node, State.SUCCEEDED, node.inputs(), output, checkpoint, null, null, null, 0);
        runs.changeRun(id, State.RUNNING, null, "CANVAS_RUN_SELECTION", Map.of("nodeRunId", nodeRunId,
                "selectedOutputId", selected.id(), "actorUserId", actor.userId()));
        return view(id, actor);
    }

    private Run control(String id, CanvasActor actor) {
        var observed = runs.run(id, false);
        runs.lockActor(actor);
        var document = documents.require(observed.canvasId(), true);
        if (!access.requireView(document, actor).canExecute() || !observed.actor().equals(actor)) {
            throw new CanvasRunException(403, "RUN_CONTROL_DENIED", "只能在原账号和团队内操作自己发起的运行");
        }
        return runs.run(id, true);
    }

    private void requireControlContext(Run run, ResourceCommandContext context) {
        if (context == null) {
            return;
        }
        if (!run.actor().equals(context.actor()) || !run.canvasId().equals(context.canvasId())) {
            throw CanvasRunException.invalid("运行控制上下文与原范围不一致");
        }
        agents.requireActive(context, run.plan().graph().projectId());
    }

    private Map<String, String> fingerprints(Plan plan) {
        return plan.nodes().stream().collect(Collectors.toMap(node -> node.node().id(), NodePlan::fingerprint));
    }

    private Map<String, String> derived(Run run, CanvasGraph graph, Map<String, CanvasNode> current,
                                        CanvasActor actor) {
        Map<String, String> result = new HashMap<>();
        var document = documents.require(run.canvasId(), false);
        for (var plan : run.plan().nodes()) {
            String id = plan.node().id();
            var node = current.get(id);
            String state = "CURRENT";
            if (node == null) {
                state = "NODE_DELETED";
            } else if (node.kind() != plan.node().kind() || !node.config()
                    .equals(plan.node().config()) || !Objects.equals(node.resource(), plan.node().resource())) {
                state = "STALE";
            } else if (!plan.action().equals("REUSE")) {
                Set<String> used = CanvasRunSourceUsage.ports(run.plan().graph(), id,
                        run.plan().nodes().stream().map(item -> item.node().id()).collect(Collectors.toSet()));
                try {
                    if (!sources.unchanged(plan, document, node, actor, used)
                            || !json.hash(environments.freeze(node, document, run.actor()))
                            .equals(json.hash(plan.environment()))) {
                        state = "STALE";
                    }
                } catch (BusinessException unavailable) {
                    state = "SOURCE_UNAVAILABLE";
                }
                var edges = graph.edges().stream()
                        .filter(edge -> edge.targetNodeId().equals(id) && edge.sourcePort() != null).toList();
                if (!new HashSet<>(edges).equals(new HashSet<>(plan.incoming()))) {
                    state = "STALE";
                }
            }
            if (state.equals("CURRENT") && plan.dependencies().stream()
                    .anyMatch(dep -> !"CURRENT".equals(result.get(dep)))) {
                state = "STALE";
            }
            result.put(id, state);
        }
        return result;
    }

    private void requestId(String id) {
        if (id == null || !id.matches("[A-Za-z0-9:._-]{1,96}")) {
            throw CanvasRunException.invalid("运行请求标识无效");
        }
    }
}
