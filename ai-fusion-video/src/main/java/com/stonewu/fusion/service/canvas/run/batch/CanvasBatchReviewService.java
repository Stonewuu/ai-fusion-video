package com.stonewu.fusion.service.canvas.run.batch;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.*;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

@Service
@RequiredArgsConstructor
public class CanvasBatchReviewService {
    public enum Action {RETRY_FAILED, USE_SUCCEEDED, FINISH}

    public record Review(@NotNull Long expectedRevision, @NotNull Action action,
                         @NotBlank @Size(max = 1000) String note, @NotNull @Valid GenerationClientScope scope) {
    }

    private final CanvasRunRepository runs;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final CanvasBatchItems batches;
    private final CanvasRunInputs inputs;
    private final CanvasRunSources sources;
    private final CanvasRunEnvironment environments;
    private final CanvasJson json;
    private final ResourceAgentFence agents;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasRun", key = "#runId")
    public void review(String runId, String nodeRunId, Review request, CanvasActor actor) {
        review(runId, nodeRunId, request, actor, null);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasRun", key = "#runId")
    public void review(String runId, String nodeRunId, Review request, CanvasActor actor,
                       ResourceCommandContext context) {
        request.scope().require(actor);
        var observed = runs.run(runId, false);
        runs.lockActor(actor);
        var document = documents.require(observed.canvasId(), true);
        access.requireExecute(document, actor);
        if (!observed.actor().equals(actor)) {
            throw new CanvasRunException(403, "RUN_CONTROL_DENIED", "只能处理自己发起的批量运行");
        }
        var run = runs.run(runId, true);
        if (context != null) {
            if (!run.actor().equals(context.actor()) || !run.canvasId().equals(context.canvasId())) {
                throw CanvasRunException.invalid("批量处理身份或画布范围不一致");
            }
            agents.requireActive(context, run.plan().graph().projectId());
        }
        var node = runs.node(nodeRunId, true);
        if (!node.runId().equals(runId) || !"BATCH".equals(node.externalKind()) || node.parentNodeRunId() != null) {
            throw CanvasRunException.invalid("此节点不是当前运行的批量任务");
        }
        if (context != null && context.agentRunId() != null) {
            for (var decision : node.checkpoint().path("decisions")) {
                if (context.agentRunId().equals(decision.path("agentRunId").asText()) && context.toolCallId()
                        .equals(decision.path("toolCallId").asText())) {
                    if (!json.hash(request).equals(decision.path("requestHash").asText())) {
                        throw CanvasRunException.conflict("同一工具调用不能改为另一个批量处理决定");
                    }
                    return;
                }
            }
        }
        if (run.cancelRequested() || run.status().terminal() || !run.deadlineAt().isAfter(Instant.now())) {
            throw CanvasRunException.fenced();
        }
        if (node.status() != State.WAITING_CONFIRMATION || node.revision() != request.expectedRevision()) {
            throw CanvasRunException.conflict("批量处理状态已变化，请刷新后重试");
        }
        if (request.note() == null || request.note().isBlank() || request.note().length() > 1000) {
            throw CanvasRunException.invalid("请填写本次处理说明（最多 1000 字）");
        }
        var all = runs.nodes(runId);
        var items = CanvasBatchItems.latest(all, node.id());
        if (items.isEmpty() || items.stream().anyMatch(item -> !item.status().terminal())) {
            throw CanvasRunException.conflict("仍有子项未结束，请先核实原任务");
        }
        var checkpoint = node.checkpoint().deepCopy();
        var decision = new LinkedHashMap<String, Object>(
                Map.of("action", request.action(), "note", request.note().strip(),
                        "actorUserId", actor.userId(), "at", Instant.now().toString(), "revision", node.revision()));
        decision.put("origin", context == null ? "user" : context.origin());
        if (context != null && context.agentRunId() != null) {
            decision.put("agentRunId", context.agentRunId());
            decision.put("toolCallId", context.toolCallId());
            decision.put("requestHash", json.hash(request));
        }
        checkpoint.withArray("decisions").add(json.object(decision));
        if (request.action() == Action.RETRY_FAILED) {
            var failed = items.stream().filter(item -> item.status() != State.SUCCEEDED).toList();
            if (failed.isEmpty()) {
                throw CanvasRunException.conflict("没有可重试的失败项");
            }
            budget(run, all, failed);
            for (var item : failed) {
                item.inputs().all().forEach(value -> sources.authorize(value, actor, true));
                var original = item.plan().environment().deepCopy();
                original.remove("businessTargets");
                if (!json.hash(environments.freeze(item.plan().node(), run.actor())).equals(json.hash(original))) {
                    throw CanvasRunException.conflict("原模型或服务配置已变化，请创建新的运行计划");
                }
            }
            batches.retry(node, failed);
            runs.updateNode(node, State.RUNNING, node.inputs(), Ports.empty(), checkpoint, "BATCH", node.id(), null, 0);
        } else {
            var output = batches.results(node, items);
            output.all().forEach(value -> sources.authorize(value, actor, true));
            if (request.action() == Action.USE_SUCCEEDED && output.all().isEmpty()) {
                throw CanvasRunException.invalid("没有可继续使用的结果，请重试失败项或结束批次");
            }
            output = inputs.publish(node, output);
            State state = request.action() == Action.USE_SUCCEEDED ? State.SUCCEEDED
                    : output.all().isEmpty() ? State.FAILED : State.PARTIAL_FAILED;
            runs.updateNode(node, state, node.inputs(), output, checkpoint, "BATCH", node.id(),
                    request.action() == Action.FINISH ? "已结束批次，保留已有结果并停止下游" : null, 0);
        }
        decision.put("nodeRunId", nodeRunId);
        runs.changeRun(runId, State.RUNNING, null, "CANVAS_RUN_BATCH_REVIEW", decision);
    }

    private void budget(Run run, List<NodeRun> all, List<NodeRun> retry) {
        var previous = all.stream().filter(node -> node.parentNodeRunId() != null && node.attempt() > 1).toList();
        int nodes = run.plan().cost().nodes()
                + previous.stream().mapToInt(node -> 1 + environments.extraAgentRuns(node.plan().node())).sum()
                + retry.stream().mapToInt(node -> 1 + environments.extraAgentRuns(node.plan().node())).sum();
        int images = run.plan().cost().images() + previous.stream().mapToInt(node -> node.plan().imageCount()).sum()
                + retry.stream().mapToInt(node -> node.plan().imageCount()).sum();
        int seconds = run.plan().cost().videoSeconds() + previous.stream().mapToInt(node -> node.plan().videoSeconds())
                .sum()
                + retry.stream().mapToInt(node -> node.plan().videoSeconds()).sum();
        var budget = run.plan().budget();
        if (nodes > budget.maxNodes() || images > budget.maxImages() || seconds > budget.maxVideoSeconds()) {
            throw CanvasRunException.invalid("重试会超出本次确认的节点、图片或视频时长上限，请结束后创建新计划");
        }
    }
}
