package com.stonewu.fusion.service.canvas.agent.node;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.canvas.CanvasAgentNodeJobMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.mapper.ai.AgentModelCallUsageMapper;
import com.stonewu.fusion.entity.ai.AgentModelCallUsage;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.time.ZoneOffset;
import java.util.List;
import java.util.Objects;

@Repository
@RequiredArgsConstructor
public class CanvasAgentNodeRepository {
    public record Job(String nodeRunId, String canvasRunId, String agentRunId, String conversationId,
                      String snapshotJson, ObjectNode input, CanvasAgentNodeScope scope, String dispatchState,
                      boolean cancelRequested, String error) {
    }

    public record Dispatch(Job job, StartedAgentRun started) {
    }

    private final CanvasAgentNodeJobMapper jobs;
    private final AgentModelCallUsageMapper usage;
    private final CanvasJson json;
    private final AgentRunRepository agents;
    private final AgentKernelSnapshotBuilder snapshots;
    private final AgentScopeV2Properties properties;

    public Job find(String nodeRunId, boolean lock) {
        var row = lock ? jobs.lock(nodeRunId) : jobs.selectById(nodeRunId);
        return row == null ? null : new Job(row.getNodeRunId(), row.getCanvasRunId(), row.getAgentRunId(),
                row.getConversationId(), row.getSnapshotJson(), json.read(row.getInputJson(), ObjectNode.class),
                json.read(row.getScopeJson(), CanvasAgentNodeScope.class), row.getDispatchState(),
                row.getCancelRequested(), row.getErrorMessage());
    }

    public void insert(String nodeRunId, String canvasRunId, String agentRunId, String conversationId,
                       CanvasAgentNodePreparation.Prepared value) {
        jobs.insertQueued(nodeRunId, canvasRunId, agentRunId, conversationId, value.snapshot().snapshotJson(),
                json.write(value.input()), json.write(value.scope()));
    }

    public List<String> pending(String instance) {
        return jobs.pending(instance);
    }

    @Transactional
    public Dispatch claim(String id, String instance) {
        var seen = find(id, false);
        if (seen == null) {
            return null;
        }
        var run = agents.lockRun(seen.agentRunId());
        var job = find(id, true);
        var now = agents.databaseNow();
        if (job.cancelRequested() || "DISPATCHED".equals(job.dispatchState()) || !"RUNNING".equals(run.getStatus())
                || run.getCancelRequestedAt() != null || !run.getDeadlineAt().isAfter(now)) {
            return null;
        }
        boolean expired = run.getLeaseUntil() == null || !run.getLeaseUntil().isAfter(now);
        if (!expired && !instance.equals(run.getOwnerInstanceId())) {
            return null;
        }
        long calls = usage.selectCount(
                new LambdaQueryWrapper<AgentModelCallUsage>().eq(AgentModelCallUsage::getRunId, run.getRunId()));
        if (run.getNextSequence() != 1 || calls != 0) {
            dispatched(id, run.getOwnerEpoch());
            return null;
        }
        if (expired) {
            run.setOwnerInstanceId(instance);
            run.setOwnerEpoch(run.getOwnerEpoch() + 1);
            var until = now.plus(properties.getExecution().getOwnerLease());
            run.setLeaseUntil(until.isAfter(run.getDeadlineAt()) ? run.getDeadlineAt() : until);
            run.setHeartbeatAt(now);
            agents.update(run);
        }
        jobs.dispatching(id);
        var snapshot = snapshots.build(json.read(job.snapshotJson(), AgentKernelSnapshotPayload.class));
        if (!snapshot.fingerprint().equals(run.getKernelFingerprint())) {
            throw CanvasRunException.conflict("专业智能体配置摘要不一致");
        }
        return new Dispatch(job,
                new StartedAgentRun(run.getRunId(), run.getConversationId(), run.getAgentStateSessionId(),
                        run.getOwnerInstanceId(), run.getOwnerEpoch(), run.getLeaseUntil().toInstant(ZoneOffset.UTC),
                        run.getDeadlineAt().toInstant(ZoneOffset.UTC), snapshot,
                        Objects.requireNonNull(agents.findInitialMessageOrder(run.getRunId()))));
    }

    public void dispatched(String id, long epoch) {
        jobs.dispatched(id, epoch);
    }

    public void scope(String id, CanvasAgentNodeScope value) {
        jobs.updateScope(json.write(value), id);
    }

    public void requestCancel(String id) {
        var job = find(id, false);
        if (job != null) {
            agents.requestInternalCancellationTree(job.agentRunId());
        }
        jobs.requestCancel(id);
    }
}
