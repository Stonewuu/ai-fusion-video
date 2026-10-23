package com.stonewu.fusion.service.canvas.run;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.service.ai.run.AgentExecutionLimits;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.entity.canvas.CanvasRunEntity;
import com.stonewu.fusion.entity.canvas.CanvasNodeRunEntity;
import com.stonewu.fusion.entity.canvas.CanvasRunPlanEntity;
import com.stonewu.fusion.mapper.canvas.CanvasRunMapper;
import com.stonewu.fusion.mapper.canvas.CanvasNodeRunMapper;
import com.stonewu.fusion.mapper.canvas.CanvasRunPlanMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.*;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

@Repository
@RequiredArgsConstructor
public class CanvasRunRepository {
    private final CanvasRunMapper runs;
    private final CanvasNodeRunMapper nodeRuns;
    private final CanvasRunPlanMapper plans;
    private final UserMapper users;
    private final CanvasJson json;
    private final GenerationOutbox outbox;
    private final AgentScopeV2Properties agentProperties;

    public record StoredPlan(Plan value, String hash) {
    }

    public void lockActor(CanvasActor actor) {
        if (users.lockEnabled(actor.userId()) == null) {
            throw new CanvasRunException(403, "ACTOR_UNAVAILABLE", "发起用户已停用");
        }
    }

    public String scope(CanvasActor actor) {
        return json.hash(actor);
    }

    public void storePlan(Plan plan) {
        plans.store(
                CanvasRunPlanEntity.builder().id(plan.id()).canvasId(plan.canvasId()).actorUserId(plan.actor().userId())
                        .actorTeamId(plan.actor().teamId()).documentRevision(plan.documentRevision())
                        .planHash(json.hash(plan))
                        .snapshotJson(json.write(plan))
                        .expiresAt(LocalDateTime.ofInstant(plan.expiresAt(), ZoneOffset.UTC)).build());
    }

    public StoredPlan plan(String id) {
        var row = plans.selectById(id);
        if (row == null) {
            throw new CanvasRunException(404, "PLAN_NOT_FOUND", "运行计划不存在，请重新预览");
        }
        return new StoredPlan(json.read(row.getSnapshotJson(), Plan.class), row.getPlanHash());
    }

    public Run run(String id, boolean lock) {
        var row = lock ? runs.lock(id) : runs.selectById(id);
        if (row == null) {
            throw new CanvasRunException(404, "RUN_NOT_FOUND", "画布运行不存在");
        }
        return run(row);
    }

    public record Receipt(String runId, String hash) {
    }

    public Receipt receipt(String requestId, CanvasActor actor) {
        var row = runs.selectOne(new LambdaQueryWrapper<CanvasRunEntity>().select(CanvasRunEntity::getId,
                        CanvasRunEntity::getRequestHash)
                .eq(CanvasRunEntity::getScopeKey, scope(actor)).eq(CanvasRunEntity::getRequestId, requestId));
        return row == null ? null : new Receipt(row.getId(), row.getRequestHash());
    }

    public boolean withdrawn(String requestId, CanvasActor actor) {
        return runs.withdrawn(scope(actor), requestId);
    }

    public void withdraw(String canvasId, String requestId, CanvasActor actor) {
        runs.withdraw(scope(actor), requestId, canvasId);
    }

    public int active(CanvasActor actor) {
        return runs.selectCount(
                        new LambdaQueryWrapper<CanvasRunEntity>().eq(CanvasRunEntity::getActorUserId, actor.userId())
                                .notIn(CanvasRunEntity::getStatus, "SUCCEEDED", "PARTIAL_FAILED", "FAILED", "CANCELLED"))
                .intValue();
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public String insert(Plan plan, Start request, ResourceCommandContext context) {
        String id = UUID.randomUUID().toString();
        Instant deadline = AgentExecutionLimits.deadlineFrom(Instant.now(),
                agentProperties.getExecution().getRunTimeout());
        runs.insertAccepted(CanvasRunEntity.builder().id(id).canvasId(plan.canvasId()).planId(plan.id())
                .actorUserId(context.actor().userId()).actorTeamId(context.actor().teamId())
                .scopeKey(scope(context.actor()))
                .requestId(request.requestId()).requestHash(json.hash(request)).contextJson(json.write(context))
                .snapshotJson(json.write(plan)).deadlineAt(LocalDateTime.ofInstant(deadline, ZoneOffset.UTC)).build());
        for (int i = 0; i < plan.nodes().size(); i++) {
            var node = plan.nodes().get(i);
            nodeRuns.insertPlanned(CanvasNodeRunEntity.builder().id(UUID.randomUUID().toString()).runId(id)
                    .nodeId(node.node().id()).ordinalNo(i).fingerprint(node.fingerprint()).planJson(json.write(node))
                    .outputJson(json.write(Ports.empty())).build());
        }
        outbox.enqueue("canvas-run:" + id + ":1", "CANVAS_RUN_ACCEPTED", id,
                Map.of("runId", id, "revision", 1, "status", "QUEUED"));
        return id;
    }

    public List<NodeRun> nodes(String runId) {
        return nodeRuns.selectList(
                        new LambdaQueryWrapper<CanvasNodeRunEntity>().eq(CanvasNodeRunEntity::getRunId, runId)
                                .orderByAsc(CanvasNodeRunEntity::getOrdinalNo, CanvasNodeRunEntity::getItemIndex,
                                        CanvasNodeRunEntity::getAttempt))
                .stream().map(this::node).toList();
    }

    public NodeRun node(String id, boolean lock) {
        var row = lock ? nodeRuns.lock(id) : nodeRuns.selectById(id);
        if (row == null) {
            throw new CanvasRunException(404, "NODE_RUN_NOT_FOUND", "节点运行不存在");
        }
        return node(row);
    }

    public NodeRun latest(String canvasId, String nodeId) {
        var row = nodeRuns.latestSucceeded(canvasId, nodeId);
        return row == null ? null : node(row);
    }

    public List<Run> history(String canvasId, int page) {
        return runs.selectPage(new Page<CanvasRunEntity>(page + 1L, 20, false),
                        new LambdaQueryWrapper<CanvasRunEntity>()
                                .eq(CanvasRunEntity::getCanvasId, canvasId)
                                .orderByDesc(CanvasRunEntity::getCreatedAt, CanvasRunEntity::getId))
                .getRecords().stream().map(this::run).toList();
    }

    public List<NodeRun> latestNodes(String canvasId) {
        return nodeRuns.latestNodes(canvasId).stream().map(this::node).toList();
    }

    public boolean hasActiveNodes(String canvasId, List<String> nodeIds) {
        return !nodeIds.isEmpty() && nodeRuns.hasActiveNodes(canvasId, nodeIds);
    }

    public List<String> activeRuns() {
        return runs.selectPage(new Page<CanvasRunEntity>(1, 64, false), new LambdaQueryWrapper<CanvasRunEntity>()
                        .select(CanvasRunEntity::getId)
                        .notIn(CanvasRunEntity::getStatus, "QUEUED", "SUCCEEDED", "PARTIAL_FAILED", "FAILED", "CANCELLED")
                        .orderByAsc(CanvasRunEntity::getUpdatedAt, CanvasRunEntity::getId))
                .getRecords().stream().map(CanvasRunEntity::getId).toList();
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public void changeRun(String id, State state, String error, String event, Map<String, Object> extra) {
        runs.changeState(state.name(), error, id);
        long revision = runs.selectOne(new LambdaQueryWrapper<CanvasRunEntity>().select(CanvasRunEntity::getRevision)
                .eq(CanvasRunEntity::getId, id)).getRevision();
        var payload = json.object(extra);
        payload.put("runId", id).put("revision", revision).put("status", state.name());
        outbox.enqueue("canvas-run:" + id + ":" + revision, event, id, payload);
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public void updateNode(NodeRun node, State state, Ports inputs, Ports outputs, ObjectNode checkpoint,
                           String kind, String externalId, String error, int delaySeconds) {
        nodeRuns.updateState(state.name(), inputs == null ? null : json.write(inputs), json.write(outputs),
                json.write(checkpoint), kind, externalId, error, delaySeconds, node.id());
        var current = run(node.runId(), false);
        changeRun(node.runId(), current.cancelRequested() ? State.CANCEL_REQUESTED : State.RUNNING,
                current.error(), "CANVAS_RUN_CHANGED", Map.of("nodeRunId", node.id(), "nodeStatus", state.name()));
    }

    public void defer(NodeRun node, int seconds) {
        nodeRuns.defer(seconds, node.id());
    }

    public Claim claim(Run run, String owner) {
        int running = nodeRuns.runningCount(run.id());
        if (running >= run.plan().budget().maxParallel()) {
            return null;
        }
        int external = nodeRuns.selectCount(
                new LambdaQueryWrapper<CanvasNodeRunEntity>().eq(CanvasNodeRunEntity::getRunId, run.id())
                        .isNotNull(CanvasNodeRunEntity::getExternalId).ne(CanvasNodeRunEntity::getExternalKind, "BATCH")
                        .notIn(CanvasNodeRunEntity::getStatus, "SUCCEEDED", "PARTIAL_FAILED", "FAILED", "BLOCKED",
                                "CANCELLED")).intValue();
        String id = nodeRuns.claimCandidate(run.id(), external, run.plan().budget().maxParallel());
        if (id == null) {
            return null;
        }
        nodeRuns.claim(owner, id);
        return new Claim(id, run.id(), owner,
                nodeRuns.selectOne(
                        new LambdaQueryWrapper<CanvasNodeRunEntity>().select(CanvasNodeRunEntity::getOwnerEpoch)
                                .eq(CanvasNodeRunEntity::getId, id)).getOwnerEpoch());
    }

    public NodeRun leased(Claim claim) {
        var row = nodeRuns.leased(claim.id(), claim.runId(), claim.owner(), claim.epoch());
        if (row == null) {
            throw CanvasRunException.fenced();
        }
        return node(row);
    }

    public boolean heartbeat(Claim claim) {
        return nodeRuns.heartbeat(claim.id(), claim.runId(), claim.owner(), claim.epoch()) == 1;
    }

    public void requestCancel(String id) {
        runs.update(new LambdaUpdateWrapper<CanvasRunEntity>().eq(CanvasRunEntity::getId, id)
                .set(CanvasRunEntity::getCancelRequested, true));
    }

    private Run run(CanvasRunEntity r) {
        return new Run(r.getId(), r.getCanvasId(),
                new CanvasActor(r.getActorUserId(), r.getActorTeamId()),
                json.read(r.getContextJson(), ResourceCommandContext.class),
                json.read(r.getSnapshotJson(), Plan.class),
                State.valueOf(r.getStatus()), r.getCancelRequested(), r.getRevision(),
                r.getErrorMsg(),
                time(r.getDeadlineAt()), time(r.getCreatedAt()), time(r.getUpdatedAt()));
    }

    private NodeRun node(CanvasNodeRunEntity r) {
        String input = r.getInputJson();
        return new NodeRun(r.getId(), r.getRunId(), r.getNodeId(),
                r.getParentNodeRunId(), r.getItemIndex(), r.getAttempt(),
                json.read(r.getPlanJson(), NodePlan.class),
                input == null ? null : json.read(input, Ports.class),
                json.read(r.getOutputJson(), Ports.class),
                json.read(r.getCheckpointJson(), ObjectNode.class), State.valueOf(r.getStatus()),
                r.getErrorMsg(),
                r.getExternalKind(), r.getExternalId(), r.getRevision(),
                r.getOwnerInstanceId(),
                r.getOwnerEpoch(), time(r.getLeaseUntil()), time(r.getUpdatedAt()));
    }

    private Instant time(LocalDateTime value) {
        return value == null ? null : value.toInstant(ZoneOffset.UTC);
    }
}
