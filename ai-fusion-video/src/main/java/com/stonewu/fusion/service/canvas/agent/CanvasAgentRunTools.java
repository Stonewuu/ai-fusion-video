package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.collaboration.CanvasPinnedOutputs;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.canvas.run.CanvasRunService;
import com.stonewu.fusion.service.canvas.run.batch.CanvasBatchReviewService;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

@Service
@RequiredArgsConstructor
public class CanvasAgentRunTools {
    private final CanvasAgentAccess access;
    private final CanvasRunService runs;
    private final CanvasRunRepository repository;
    private final CanvasJson json;
    private final CanvasBatchReviewService batches;
    private final CanvasPinnedOutputs fixed;

    public Object fixedOutput(String edgeId, int offset, int limit, ToolExecutionContext tool) {
        var session = access.require(tool);
        var value = fixed.edge(session.scope().canvasId(), edgeId, session.context().actor());
        return Map.of("status", "success", "content", CanvasAgentLibrary.page(json.write(value), offset, limit));
    }

    public Object plan(long revision, String mode, List<String> nodeIds, Map<String, String> reuse, Budget budget,
                       ToolExecutionContext tool) {
        var session = access.require(tool);
        var plan = runs.plan(session.scope().canvasId(),
                new PlanRequest(revision, Scope.valueOf(mode), nodeIds, reuse, budget, scope(session)),
                session.context().actor());
        return Map.of("status", plan.issues().isEmpty() ? "ready" : "blocked", "plan",
                Map.of("id", plan.id(), "hash", plan.hash(),
                        "documentRevision", plan.documentRevision(), "cost", plan.cost(), "issues",
                        plan.issues().stream().limit(100).toList(), "expiresAt", plan.expiresAt(),
                        "totalIssues", plan.issues().size(),
                        "totalNodes", plan.nodes().size(), "nodes",
                        plan.nodes().stream().limit(100).map(node -> Map.of("nodeId", node.nodeId(),
                                "title", node.title(), "action", node.action(), "kind", node.kind())).toList()));
    }

    public Object output(String id, String nodeRunId, String outputId, int offset, int limit,
                         ToolExecutionContext tool) {
        var node = node(id, nodeRunId, tool);
        var values = new ArrayList<>(node.outputs().all());
        if (node.status() == State.WAITING_SELECTION) {
            values.addAll(node.inputs().at("candidates"));
        }
        var output = values.stream().filter(value -> value.id().equals(outputId)).findFirst()
                .orElseThrow(() -> new BusinessException(404, "此节点没有该产物或已失去访问权限"));
        return Map.of("status", "success", "content", CanvasAgentLibrary.page(json.write(output), offset, limit));
    }

    public Object outputs(String id, String nodeRunId, String port, int offset, int limit, ToolExecutionContext tool) {
        if (offset < 0 || limit < 1 || limit > 50) {
            throw new BusinessException(400, "产物分页范围无效");
        }
        var node = node(id, nodeRunId, tool);
        var values = port.equals("candidates") && node.status() == State.WAITING_SELECTION ? node.inputs()
                .at(port) : node.outputs().at(port);
        return Map.of("status", "success", "total", values.size(), "offset", offset, "hasMore",
                (long) offset + limit < values.size(),
                "items", values.stream().skip(offset).limit(limit).map(CanvasAgentRunTools::value).toList());
    }

    public Object writeback(String id, String nodeRunId, int offset, int limit, ToolExecutionContext tool) {
        var proposal = node(id, nodeRunId, tool).writeback();
        if (proposal == null) {
            throw new BusinessException(404, "此节点没有已固定的写回提案");
        }
        return Map.of("status", "success", "content", CanvasAgentLibrary.page(json.write(proposal), offset, limit));
    }

    private NodeView node(String id, String nodeRunId, ToolExecutionContext tool) {
        var session = access.require(tool);
        requireBound(id, session);
        return runs.view(id, session.context().actor()).nodes().stream()
                .filter(item -> item.id().equals(nodeRunId) && !item.contentUnavailable()).findFirst()
                .orElseThrow(() -> new BusinessException(404, "节点运行不存在或已失去访问权限"));
    }

    public Object start(String planId, String planHash, ToolExecutionContext tool) {
        var session = access.require(tool);
        var run = runs.start(session.scope().canvasId(),
                new Start(session.requestId(), planId, planHash, scope(session)), session.context());
        return Map.of("status", "accepted", "canvasRunId", run.id(), "run", summary(run, 0, 20),
                "nextAction", "使用 await_canvas_run 等待结果；不要重复启动同一任务");
    }

    public Object get(String id, int offset, int limit, ToolExecutionContext tool) {
        var session = access.require(tool);
        requireBound(id, session);
        return summary(runs.view(id, session.context().actor()), offset, limit);
    }

    public Object cancel(String id, ToolExecutionContext tool) {
        var session = access.require(tool);
        requireBound(id, session);
        return summary(runs.cancel(id, scope(session), session.context().actor(), session.context()), 0, 20);
    }

    public Object select(String id, String nodeRunId, long revision, String outputId, ToolExecutionContext tool) {
        var session = access.require(tool);
        requireBound(id, session);
        return summary(
                runs.select(id, nodeRunId, new Selection(revision, outputId, scope(session)), session.context().actor(),
                        session.context()), 0, 20);
    }

    public Object reviewBatch(String id, String nodeRunId, long revision, String action, String note,
                              ToolExecutionContext tool) {
        var session = access.require(tool);
        requireBound(id, session);
        batches.review(id, nodeRunId,
                new CanvasBatchReviewService.Review(revision, CanvasBatchReviewService.Action.valueOf(action), note,
                        scope(session)),
                session.context().actor(), session.context());
        return summary(runs.view(id, session.context().actor()), 0, 20);
    }

    public void requireBound(String id, CanvasAgentAccess.Session session) {
        if (!repository.run(id, false).canvasId().equals(session.scope().canvasId())) {
            throw new BusinessException(403, "运行不属于本轮绑定的画布");
        }
    }

    public static GenerationClientScope scope(CanvasAgentAccess.Session session) {
        return new GenerationClientScope(session.context().actor().userId(), session.context().actor().teamId());
    }

    public static Map<String, Object> summary(View run, int offset, int limit) {
        if (offset < 0 || limit < 1 || limit > 50) {
            throw new BusinessException(400, "运行结果每页最多 50 个节点");
        }
        var result = new LinkedHashMap<String, Object>();
        result.put("status", run.status());
        result.put("canvasRunId", run.id());
        result.put("canvasId", run.canvasId());
        result.put("revision", run.revision());
        result.put("canControl", run.canControl());
        result.put("error", run.error());
        result.put("statistics", run.statistics());
        result.put("totalNodes", run.nodes().size());
        result.put("offset", offset);
        result.put("hasMore", (long) offset + limit < run.nodes().size());
        result.put("nodes", run.nodes().stream().skip(offset).limit(limit).map(node -> {
            var value = new LinkedHashMap<String, Object>();
            value.put("nodeRunId", node.id());
            value.put("nodeId", node.nodeId());
            value.put("title", node.title());
            value.put("parentNodeRunId", node.parentNodeRunId());
            value.put("itemIndex", node.itemIndex());
            value.put("attempt", node.attempt());
            value.put("externalKind", node.externalKind());
            value.put("status", node.status());
            value.put("revision", node.revision());
            value.put("error", node.error());
            value.put("contentUnavailable", node.contentUnavailable());
            value.put("hasWriteback", node.writeback() != null);
            if (node.writeback() != null) {
                value.put("writeback", writebackSummary(node.writeback()));
            }
            var outputs = new LinkedHashMap<String, Object>();
            node.outputs().values().forEach((port, items) -> outputs.put(port, Map.of("total", items.size(),
                    "items", items.stream().limit(6).map(CanvasAgentRunTools::value).toList(), "hasMore",
                    items.size() > 6)));
            value.put("outputs", outputs);
            if (node.status() == State.WAITING_SELECTION) {
                value.put("candidates",
                        node.inputs().at("candidates").stream().limit(6).map(CanvasAgentRunTools::value).toList());
                value.put("totalCandidates", node.inputs().at("candidates").size());
            }
            return value;
        }).toList());
        return result;
    }

    private static Object value(Value item) {
        var value = new LinkedHashMap<String, Object>();
        value.put("id", item.id());
        value.put("type", item.type());
        value.put("mediaId", item.mediaId());
        value.put("mediaVersion", item.mediaVersion());
        value.put("resource", item.resource());
        if (item.text() != null) {
            value.put("excerpt", item.text().substring(0, Math.min(160, item.text().length())));
            value.put("totalCharacters", item.text().length());
        }
        return value;
    }

    private static Object writebackSummary(ObjectNode proposal) {
        var summary = new LinkedHashMap<String, Object>();
        summary.put("hash", proposal.path("hash"));
        summary.put("binding", proposal.path("binding"));
        summary.put("target", proposal.path("target").path("resource"));
        summary.put("dangerous", proposal.path("dangerous").asBoolean());
        summary.put("notice", proposal.path("notice"));
        summary.put("expectedParentVersion", proposal.path("expectedParentVersion"));
        summary.put("decided", proposal.hasNonNull("decision"));
        return summary;
    }
}
