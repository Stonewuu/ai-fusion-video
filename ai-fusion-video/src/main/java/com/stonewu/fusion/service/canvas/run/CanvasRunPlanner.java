package com.stonewu.fusion.service.canvas.run;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.CanvasGraphValidator;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeInputs;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeTypes;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.collaboration.CanvasPinnedOutputs;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpPlanner;
import com.stonewu.fusion.service.canvas.run.writeback.CanvasWritebackPreview;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.*;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

@Component
@RequiredArgsConstructor
public class CanvasRunPlanner {
    private final CanvasGraphValidator validator;
    private final CanvasCapabilityRegistry capabilities;
    private final CanvasRunSources sources;
    private final CanvasRunEnvironment environments;
    private final CanvasRunRepository runs;
    private final CanvasRunExecutors executors;
    private final CanvasJson json;
    private final CanvasWritebackPreview writebacks;
    private final CanvasAgentNodeInputs agentInputs;
    private final CanvasMcpPlanner mcp;
    private final CanvasPinnedOutputs fixedOutputs;

    public Plan build(CanvasDocument document, CanvasGraph graph, PlanRequest request, CanvasActor actor) {
        bounds(request);
        var order = validator.validate(graph).topologicalNodeIds();
        Map<String, CanvasNode> nodes = new LinkedHashMap<>();
        graph.nodes().forEach(node -> nodes.put(node.id(), node));
        if (!nodes.keySet().containsAll(request.nodeIds()) || !nodes.keySet().containsAll(request.reuse().keySet())) {
            throw CanvasRunException.invalid("选中的节点或复用节点已不在画布中");
        }
        if (request.mode() == Scope.SINGLE && request.nodeIds().size() != 1) {
            throw CanvasRunException.invalid("单节点运行必须且只能选中一个节点");
        }
        Set<String> selected = new LinkedHashSet<>(request.mode() == Scope.ALL || request.mode() == Scope.STALE
                ? nodes.keySet() : request.nodeIds());
        selected.removeIf(id -> organization(nodes.get(id)));
        if (request.mode() == Scope.DEPENDENCIES || request.mode() == Scope.STALE) {
            expand(selected, graph, request.reuse());
        }
        // 单节点和选中子图不会偷偷执行其余上游；显式复用上游才把该固定产物纳入计划。
        for (var edge : graph.edges()) {
            if (execution(edge) && selected.contains(edge.targetNodeId())
                    && request.reuse().containsKey(edge.sourceNodeId())) {
                selected.add(edge.sourceNodeId());
            }
        }
        if (selected.size() > 1000) {
            throw CanvasRunException.invalid("单次运行最多包含 1000 个节点，请缩小范围");
        }
        List<Issue> issues = new ArrayList<>();
        Map<String, NodePlan> plans = new LinkedHashMap<>();
        for (String id : order) {
            if (!selected.contains(id)) {
                continue;
            }
            var node = nodes.get(id);
            try {
                plans.put(id, node(document, graph, node, request, actor, plans, selected));
            } catch (BusinessException failure) {
                if (failure.getCode() == 403) {
                    throw failure;
                }
                issues.add(new Issue(id, "NODE_INVALID", failure.getMessage()));
            }
        }
        if (request.mode() == Scope.STALE && issues.isEmpty()) {
            Set<String> stale = new LinkedHashSet<>();
            plans.forEach((id, plan) -> {
                var latest = runs.latest(document.id(), id);
                if (latest == null || !latest.plan().fingerprint().equals(plan.fingerprint())) {
                    stale.add(id);
                }
            });
            expand(stale, graph, request.reuse());
            plans.keySet().removeIf(id -> !stale.contains(id));
        }
        int batchItems = plans.values().stream().filter(plan -> !plan.action().equals("REUSE"))
                .mapToInt(plan -> environments.batchItems(plan.node())).sum();
        int agentRuns = plans.values().stream().filter(plan -> !plan.action().equals("REUSE"))
                .mapToInt(plan -> environments.agentRuns(plan.node())).sum();
        int extraAgents = plans.values().stream().filter(plan -> !plan.action().equals("REUSE"))
                .mapToInt(plan -> environments.extraAgentRuns(plan.node())).sum();
        var cost = new Cost(plans.size() + batchItems + extraAgents,
                plans.values().stream().mapToInt(NodePlan::imageCount).sum(),
                plans.values().stream().mapToInt(NodePlan::videoSeconds).sum(), batchItems, agentRuns,
                plans.values().stream().filter(plan -> !plan.action().equals("REUSE"))
                        .mapToInt(plan -> environments.mcpCalls(plan.node())).sum(), "未配置");
        if (selected.isEmpty() || plans.isEmpty()) {
            issues.add(new Issue(null, "EMPTY_SCOPE", "当前范围没有需要执行的节点"));
        }
        if (cost.nodes() > request.budget().maxNodes() || cost.images() > request.budget().maxImages()
                || cost.videoSeconds() > request.budget().maxVideoSeconds() || cost.batchItems() > request.budget()
                .maxBatchItems()) {
            issues.add(new Issue(null, "BUDGET_EXCEEDED", "运行数量超过本次设置的上限，请调整范围或预算"));
        }
        for (var plan : plans.values()) {
            validateInputs(plan, plans, issues);
        }
        return new Plan(UUID.randomUUID().toString(), document.id(), document.revision(), request.mode(),
                List.copyOf(request.nodeIds()),
                actor, graph, List.copyOf(plans.values()), request.budget(), cost, List.copyOf(issues),
                Instant.now().plusSeconds(900));
    }

    private NodePlan node(CanvasDocument document, CanvasGraph graph, CanvasNode node, PlanRequest request,
                          CanvasActor actor, Map<String, NodePlan> previous, Set<String> selected) {
        List<CanvasEdge> incoming = graph.edges().stream()
                .filter(edge -> execution(edge) && edge.targetNodeId().equals(node.id()))
                .sorted(Comparator.comparing(CanvasEdge::targetPort).thenComparing(CanvasEdge::inputIndex)).toList();
        Set<String> used = CanvasRunSourceUsage.ports(graph, node.id(), selected);
        String reused = request.reuse().get(node.id());
        if (reused != null) {
            var original = reusable(reused, document.id(), node.id(), actor);
            var output = original.outputs();
            return new NodePlan(node, json.hash(Map.of("reuse", reused, "outputs", output)), "REUSE", reused,
                    List.of(), List.of(), Map.of(), output, json.object(Map.of()), 0, 0);
        }
        if (!executors.supports(node.kind())) {
            throw CanvasRunException.invalid("此节点的执行能力尚未接通：" + capabilities.require(node.kind()).label());
        }
        Map<String, Integer> observedOutputs = new HashMap<>();
        for (var upstream : previous.values()) {
            if (!upstream.action().equals("REUSE") && upstream.sourceOutputs().values().isEmpty()) {
                continue;
            }
            for (var port : capabilities.require(upstream.node().kind()).outputs()) {
                observedOutputs.put(upstream.node().id() + ":" + port.id(),
                        upstream.sourceOutputs().at(port.id()).size());
            }
        }
        environments.validateForRun(graph, node, observedOutputs);
        var environment = environments.freeze(node, document, actor);
        var output = sources.freeze(document, node, actor, used);
        Map<String, Value> pinned = new LinkedHashMap<>();
        List<String> dependencies = new ArrayList<>();
        List<Object> bindings = new ArrayList<>();
        for (var edge : incoming) {
            if (edge.output() != null) {
                var value = fixedOutputs.resolve(document.id(), edge, actor);
                pinned.put(edge.id(), value);
                bindings.add(List.of(edge.targetPort(), edge.inputIndex(), value));
            } else {
                if (!selected.contains(edge.sourceNodeId())) {
                    throw CanvasRunException.invalid(
                            "输入“" + edge.targetPort() + "”来自未运行的节点；请选择包含依赖，或固定历史输出");
                }
                if (!dependencies.contains(edge.sourceNodeId())) {
                    dependencies.add(edge.sourceNodeId());
                }
                var upstream = previous.get(edge.sourceNodeId());
                bindings.add(List.of(edge.targetPort(), edge.inputIndex(), edge.sourcePort(),
                        upstream == null ? "invalid" : upstream.fingerprint()));
            }
        }
        String fingerprint = json.hash(Map.of("kind", node.kind(), "config", node.config(), "sources", output,
                "environment", environment, "bindings", bindings));
        if (node.kind() == CanvasNodeKind.WRITEBACK) {
            List<Value> targets = new ArrayList<>();
            for (var edge : incoming) {
                if (edge.targetPort().equals("target")) {
                    if (pinned.containsKey(edge.id())) {
                        targets.add(pinned.get(edge.id()));
                    } else if (previous.containsKey(edge.sourceNodeId())) {
                        targets.addAll(previous.get(edge.sourceNodeId()).sourceOutputs().at(edge.sourcePort()));
                    }
                }
            }
            writebacks.plan(document, targets, actor);
        }
        return new NodePlan(node, fingerprint, "EXECUTE", null, List.copyOf(dependencies), incoming, Map.copyOf(pinned),
                output,
                environment, environments.images(node), environments.seconds(node, environment));
    }

    private NodeRun reusable(String id, String canvasId, String nodeId, CanvasActor actor) {
        var original = runs.node(id, false);
        if (!runs.run(original.runId(), false).canvasId().equals(canvasId) || !original.nodeId().equals(nodeId)
                || original.status() != State.SUCCEEDED || original.parentNodeRunId() != null) {
            throw CanvasRunException.invalid("只能固定本画布对应节点的一次成功输出；批量子项请使用汇总结果");
        }
        original.outputs().all().forEach(value -> sources.authorize(value, actor, true));
        return original;
    }

    private void validateInputs(NodePlan plan, Map<String, NodePlan> nodes, List<Issue> issues) {
        if (plan.action().equals("REUSE")) {
            return;
        }
        for (var port : capabilities.require(plan.node().kind()).inputs()) {
            var connected = plan.incoming().stream().filter(edge -> edge.targetPort().equals(port.id())).toList();
            var literal = plan.node().config().get(port.id());
            boolean hasContext = port.id().equals("prompt") && plan.incoming().stream()
                    .anyMatch(edge -> edge.targetPort().equals("context"));
            boolean optionalPrompt = port.id().equals("prompt") && CanvasRunInputs.optionalVideoPrompt(plan);
            if (port.required() && !optionalPrompt && !hasContext && connected.isEmpty() && (literal == null || literal.isNull() || literal.asText()
                    .isBlank())) {
                issues.add(new Issue(plan.node().id(), "MISSING_INPUT", "请填写或连接“" + port.label() + "”"));
            }
            for (var edge : connected) {
                var upstream = nodes.get(edge.sourceNodeId());
                if (edge.output() == null && upstream == null) {
                    issues.add(new Issue(plan.node().id(), "INVALID_UPSTREAM", "上游节点未通过校验"));
                } else if (edge.output() == null && (upstream.action().equals("REUSE")
                        || !upstream.sourceOutputs().values().isEmpty()) && upstream.sourceOutputs()
                        .at(edge.sourcePort()).isEmpty()) {
                    issues.add(new Issue(plan.node().id(), "EMPTY_SOURCE",
                            "上游“" + upstream.node().title() + "”没有“" + edge.sourcePort() + "”输出"));
                }
            }
        }
        var operation = plan.node().kind() == CanvasNodeKind.FOREACH ? environments.batchBody(
                plan.node()) : plan.node();
        if (CanvasAgentNodeTypes.KINDS.contains(operation.kind())) {
            try {
                agentInputs.plan(operation, plan, nodes, plan.node().kind() == CanvasNodeKind.FOREACH);
            } catch (BusinessException failure) {
                issues.add(new Issue(plan.node().id(), "AGENT_INPUT_INVALID", failure.getMessage()));
            }
        }
        if (operation.kind() == CanvasNodeKind.MCP_TOOL) {
            try {
                mcp.validate(operation, plan, nodes, plan.node().kind() == CanvasNodeKind.FOREACH);
            } catch (BusinessException failure) {
                issues.add(new Issue(plan.node().id(), "MCP_INPUT_INVALID", failure.getMessage()));
            }
        }
    }

    private void expand(Set<String> selected, CanvasGraph graph, Map<String, String> reuse) {
        Map<String, List<String>> dependencies = new HashMap<>();
        for (var edge : graph.edges()) {
            if (execution(edge) && edge.output() == null) {
                dependencies.computeIfAbsent(edge.targetNodeId(), ignored -> new ArrayList<>())
                        .add(edge.sourceNodeId());
            }
        }
        var queue = new ArrayDeque<>(selected);
        while (!queue.isEmpty()) {
            String current = queue.removeFirst();
            if (reuse.containsKey(current)) {
                continue;
            }
            for (String source : dependencies.getOrDefault(current, List.of())) {
                if (selected.add(source)) {
                    queue.addLast(source);
                }
            }
        }
    }

    private void bounds(PlanRequest request) {
        if (request == null || request.mode() == null || request.budget() == null || request.nodeIds() == null || request.reuse() == null) {
            throw CanvasRunException.invalid("缺少运行范围或预算");
        }
        var b = request.budget();
        if (b.maxNodes() < 1 || b.maxNodes() > 1000 || b.maxImages() < 0 || b.maxImages() > 1000
                || b.maxVideoSeconds() < 0 || b.maxVideoSeconds() > 36000 || b.maxParallel() < 1 || b.maxParallel() > 8
                || b.maxBatchItems() < 1 || b.maxBatchItems() > 1000 || request.nodeIds()
                .size() > 1000 || request.reuse().size() > 1000) {
            throw CanvasRunException.invalid("节点、图片、视频时长、并发或批量上限无效");
        }
    }

    private boolean execution(CanvasEdge edge) {
        return edge.kind() == CanvasEdgeKind.EXECUTION;
    }

    private boolean organization(CanvasNode node) {
        return node.kind() == CanvasNodeKind.NOTE || node.kind() == CanvasNodeKind.GROUP;
    }

    public PlanView view(Plan plan) {
        return new PlanView(plan.id(), plan.canvasId(), plan.documentRevision(), json.hash(plan), plan.mode(),
                plan.nodes().stream()
                        .map(item -> new PlanNode(item.node().id(), item.node().title(), item.node().kind().code(),
                                item.action(), item.reusedNodeRunId(), item.dependencies(), item.imageCount(),
                                item.videoSeconds(), mcp.describe(item))).toList(),
                plan.budget(), plan.cost(), plan.issues(), plan.expiresAt());
    }
}
