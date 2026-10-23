package com.stonewu.fusion.service.canvas.run;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.run.batch.CanvasBatchItems;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

/**
 * 短事务只做调度、持久化受理和状态观察；耗时外部工作由对应持久化任务执行。
 */
@Service
@RequiredArgsConstructor
public class CanvasRunTransactions {
    private final CanvasRunRepository runs;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final CanvasRunInputs inputs;
    private final CanvasRunSources sources;
    private final CanvasRunExecutors executors;
    private final CanvasRunOrphanSettlement orphanSettlement;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasRun", key = "#runId")
    public Claim schedule(String runId, String owner) {
        Run run = lock(runId, true);
        if (run.status().terminal() || run.status() == State.QUEUED) {
            return null;
        }
        if (!run.cancelRequested() && !run.deadlineAt().isAfter(Instant.now())) {
            runs.requestCancel(run.id());
            runs.changeRun(run.id(), State.CANCEL_REQUESTED,
                    "运行已超过配置的执行时限，停止后续节点并请求取消已提交任务", "CANVAS_RUN_CANCELLED", Map.of());
            run = runs.run(run.id(), false);
        }
        var nodes = runs.nodes(run.id());
        var indexed = nodes.stream().filter(node -> node.parentNodeRunId() == null)
                .collect(Collectors.toMap(NodeRun::nodeId, Function.identity()));
        for (var node : nodes) {
            if (node.status().terminal()) {
                continue;
            }
            if (run.cancelRequested()) {
                if (node.externalId() != null) {
                    executors.require(node.plan().node().kind()).cancel(run, node);
                    if (node.status() != State.CANCEL_REQUESTED && node.status() != State.NEEDS_REVIEW && node.status() != State.UNKNOWN_SUBMISSION) {
                        update(node, State.CANCEL_REQUESTED, node.inputs(), node.outputs(), null);
                    }
                } else {
                    update(node, State.CANCELLED, node.inputs(), node.outputs(), null);
                }
                continue;
            }
            if (node.status() != State.WAITING_INPUT) {
                continue;
            }
            var deps = node.plan().dependencies().stream().map(indexed::get).toList();
            if (deps.stream()
                    .anyMatch(dep -> dep == null || (dep.status().terminal() && dep.status() != State.SUCCEEDED))) {
                update(node, State.BLOCKED, null, Ports.empty(), "上游执行未成功，此分支已停止");
            } else if (deps.stream().allMatch(dep -> dep.status() == State.SUCCEEDED)) {
                try {
                    update(node, State.READY, inputs.resolve(node.plan(), indexed, run.actor()), Ports.empty(), null);
                } catch (BusinessException failure) {
                    update(node, State.FAILED, null, Ports.empty(), failure.getMessage());
                }
            }
        }
        summarize(runs.run(run.id(), false), runs.nodes(run.id()));
        var current = runs.run(run.id(), false);
        return current.status().terminal() ? null : runs.claim(current, owner);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED, rollbackFor = Exception.class)
    @CacheEvict(value = "canvasRun", key = "#claim.runId()")
    public void step(Claim claim) throws Exception {
        var run = lock(claim.runId(), true);
        var node = runs.leased(claim);
        if (run.cancelRequested() && node.externalId() == null) {
            update(node, State.CANCELLED, node.inputs(), node.outputs(), null);
            return;
        }
        if (!run.cancelRequested()) {
            node.inputs().all().forEach(value -> sources.authorize(value, run.actor(), true));
            node.plan().sourceOutputs().all().forEach(value -> sources.authorize(value, run.actor(), true));
        }
        var outcome = node.plan().action().equals("REUSE")
                ? CanvasRunExecutors.Outcome.done(node.plan().sourceOutputs())
                : executors.require(node.plan().node().kind()).step(run, node, claim);
        State state = outcome.state();
        Ports output = outcome.outputs();
        if (state.terminal() && run.cancelRequested()) {
            state = State.CANCELLED;
            output = Ports.empty();
        } else if (state == State.SUCCEEDED || state == State.PARTIAL_FAILED) {
            output.all().forEach(value -> sources.authorize(value, run.actor(), true));
            output = inputs.publish(node, output);
        }
        if (state == node.status() && node.outputs().equals(output) && Objects.equals(node.error(), outcome.error())
                && Objects.equals(node.externalId(), outcome.externalId())) {
            runs.defer(node, 2);
        } else {
            runs.updateNode(node, state, node.inputs(), output, node.checkpoint(),
                    outcome.externalKind(), outcome.externalId(), outcome.error(), 2);
        }
        summarize(runs.run(run.id(), false), runs.nodes(run.id()));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasRun", key = "#claim.runId()")
    public void failed(Claim claim, String error) {
        var run = lock(claim.runId(), true);
        var node = runs.leased(claim);
        // 已有持久化副作用不能误判为普通失败，保留关联并继续观察原任务。
        if (node.externalId() != null) {
            if (node.status() == State.NEEDS_REVIEW && error.equals(node.error())) {
                runs.defer(node, 10);
            } else {
                update(node, State.NEEDS_REVIEW, node.inputs(), node.outputs(), error);
            }
        } else {
            update(node, run.cancelRequested() ? State.CANCELLED : State.FAILED, node.inputs(), node.outputs(), error);
        }
        summarize(runs.run(run.id(), false), runs.nodes(run.id()));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasRun", key = "#runId")
    public void inaccessible(String runId) {
        // 授权已撤销后只停止此运行的后续调度，不伪造第三方任务的停止结果。
        var run = runs.run(runId, true);
        if (run.status().terminal()) {
            return;
        }
        runs.requestCancel(runId);
        for (var node : runs.nodes(runId)) {
            if (node.status().terminal() || "BATCH".equals(node.externalKind())) {
                continue;
            }
            if (node.externalId() == null || orphanSettlement.ended(node)) {
                update(node, State.CANCELLED, node.inputs(), Ports.empty(), "运行授权已失效，后续执行已停止");
            } else if (node.status() != State.NEEDS_REVIEW) {
                update(node, State.NEEDS_REVIEW, node.inputs(), node.outputs(),
                        "运行授权已失效，原外部任务需要管理员核实");
            }
        }
        var settled = runs.nodes(runId);
        for (var node : settled) {
            if (node.status().terminal() || !"BATCH".equals(node.externalKind())) {
                continue;
            }
            if (node.id().equals(node.externalId()) && CanvasBatchItems.latest(settled, node.id()).stream()
                    .allMatch(item -> item.status().terminal())) {
                update(node, State.CANCELLED, node.inputs(), Ports.empty(), "运行授权已失效，批量子项已结束");
            } else if (node.status() != State.NEEDS_REVIEW) {
                update(node, State.NEEDS_REVIEW, node.inputs(), Ports.empty(), "运行授权已失效，等待原批量子项核实");
            }
        }
        var current = runs.run(runId, false);
        summarize(current, runs.nodes(runId));
    }

    private Run lock(String id, boolean permitCancelled) {
        var observed = runs.run(id, false);
        runs.lockActor(observed.actor());
        var document = documents.require(observed.canvasId(), true);
        // 归档后的取消与已提交任务观察仍可继续；不能再创建新的副作用。
        access.requireView(document, observed.actor());
        var run = runs.run(id, true);
        if (document.archived() && !run.cancelRequested() && !run.status().terminal()) {
            runs.requestCancel(id);
            runs.changeRun(id, State.CANCEL_REQUESTED, "画布已归档，停止后续节点", "CANVAS_RUN_CANCELLED", Map.of());
            run = runs.run(id, false);
        }
        if (!run.cancelRequested()) {
            access.requireExecute(document, run.actor());
        }
        if (!permitCancelled && (run.cancelRequested() || !run.deadlineAt().isAfter(Instant.now()))) {
            throw CanvasRunException.fenced();
        }
        return run;
    }

    private void update(NodeRun node, State state, Ports input, Ports output, String error) {
        runs.updateNode(node, state, input, output, node.checkpoint(), node.externalKind(), node.externalId(), error,
                0);
    }

    private void summarize(Run run, List<NodeRun> nodes) {
        var roots = nodes.stream().filter(node -> node.parentNodeRunId() == null).toList();
        // 历史失败 attempt 保留在运行记录中，不再决定当前批次的终态。
        var active = new ArrayList<>(roots);
        roots.stream().filter(node -> "BATCH".equals(node.externalKind()))
                .forEach(node -> active.addAll(CanvasBatchItems.latest(nodes, node.id())));
        State state;
        if (roots.stream().allMatch(node -> node.status().terminal())) {
            if (run.cancelRequested()) {
                state = State.CANCELLED;
            } else if (roots.stream().allMatch(node -> node.status() == State.SUCCEEDED)) {
                state = State.SUCCEEDED;
            } else {
                state = roots.stream().anyMatch(
                        node -> node.status() == State.SUCCEEDED || node.status() == State.PARTIAL_FAILED) ? State.PARTIAL_FAILED : State.FAILED;
            }
        } else if (active.stream().anyMatch(node -> node.status() == State.UNKNOWN_SUBMISSION)) {
            state = State.UNKNOWN_SUBMISSION;
        } else if (active.stream().anyMatch(node -> node.status() == State.NEEDS_REVIEW)) {
            state = State.NEEDS_REVIEW;
        } else if (run.cancelRequested()) {
            state = State.CANCEL_REQUESTED;
        } else if (nodes.stream().anyMatch(node -> node.status() == State.WAITING_SELECTION)) {
            state = State.WAITING_SELECTION;
        } else if (nodes.stream().anyMatch(node -> node.status() == State.WAITING_CONFIRMATION)) {
            state = State.WAITING_CONFIRMATION;
        } else {
            state = State.RUNNING;
        }
        if (run.status() != state) {
            runs.changeRun(run.id(), state, run.error(), "CANVAS_RUN_CHANGED", Map.of());
        }
    }
}
