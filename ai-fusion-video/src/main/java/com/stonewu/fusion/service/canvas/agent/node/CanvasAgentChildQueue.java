package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScope;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.canvas.CanvasAgentNodeJobMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.mapper.ai.AgentModelCallUsageMapper;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.entity.ai.AgentModelCallUsage;
import com.stonewu.fusion.entity.ai.AgentMessage;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.time.ZoneOffset;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * 子任务与初始消息已经同事务受理；只恢复尚无执行事件、尚无模型调用的启动缺口。
 */
@Repository
@RequiredArgsConstructor
public class CanvasAgentChildQueue {
    public record Dispatch(StartedAgentRun started, AgentRun child, AgentRun parent, CanvasActor actor,
                           String canvasId, String input) {
    }

    private final AgentRunRepository agents;
    private final CanvasAgentNodeRepository nodes;
    private final AgentKernelSnapshotBuilder snapshots;
    private final AgentScopeV2Properties properties;
    private final CanvasAgentNodeJobMapper jobs;
    private final AgentModelCallUsageMapper usage;
    private final AgentMessageMapper messages;
    private final CanvasJson json;

    public List<String> pending(String instance) {
        return jobs.pendingChildren(instance);
    }

    @Transactional
    public Dispatch claim(String id, String instance) {
        var child = agents.lockRun(id);
        var now = agents.databaseNow();
        if (child == null || child.getParentRunId() == null || !"RUNNING".equals(
                child.getStatus()) || child.getNextSequence() != 1
                || child.getCanvasScopeJson() == null || child.getCancelRequestedAt() != null || !child.getDeadlineAt()
                .isAfter(now)) {
            return null;
        }
        var scope = json.read(child.getCanvasScopeJson(), CanvasAgentScope.class);
        var job = scope.nodeRunId() == null ? null : nodes.find(scope.nodeRunId(), false);
        var parent = agents.findRun(child.getParentRunId());
        if (job == null || job.cancelRequested() || parent == null || parent.getCancelRequestedAt() != null
                || !Set.of("RUNNING", "WAITING_EXTERNAL", "WAITING_CONFIRMATION").contains(parent.getStatus())
                || !parent.getDeadlineAt().isAfter(now) || !Objects.equals(child.getCanvasScopeJson(),
                parent.getCanvasScopeJson())) {
            return null;
        }
        boolean expired = child.getLeaseUntil() == null || !child.getLeaseUntil().isAfter(now);
        if (!expired && !instance.equals(child.getOwnerInstanceId())) {
            return null;
        }
        if (usage.selectCount(
                new LambdaQueryWrapper<AgentModelCallUsage>().eq(AgentModelCallUsage::getRunId, id)) != 0) {
            return null;
        }
        if (expired) {
            child.setOwnerInstanceId(instance);
            child.setOwnerEpoch(child.getOwnerEpoch() + 1);
            var lease = now.plus(properties.getExecution().getOwnerLease());
            child.setLeaseUntil(lease.isAfter(child.getDeadlineAt()) ? child.getDeadlineAt() : lease);
            child.setHeartbeatAt(now);
            agents.update(child);
        }
        var snapshot = snapshots.build(
                json.read(child.getAgentDefinitionSnapshotJson(), AgentKernelSnapshotPayload.class));
        if (!snapshot.fingerprint().equals(child.getKernelFingerprint())) {
            throw new IllegalStateException("子任务配置摘要不一致");
        }
        Long order = agents.findInitialMessageOrder(id);
        if (order == null) {
            throw new IllegalStateException("子任务缺少已受理的初始消息");
        }
        String input = messages.selectOne(
                new LambdaQueryWrapper<AgentMessage>().select(AgentMessage::getId, AgentMessage::getContent)
                        .eq(AgentMessage::getRunId, id).eq(AgentMessage::getMessageOrder, order)
                        .eq(AgentMessage::getRole, "user")).getContent();
        var started = new StartedAgentRun(id, child.getConversationId(), child.getAgentStateSessionId(),
                child.getOwnerInstanceId(), child.getOwnerEpoch(),
                child.getLeaseUntil().toInstant(ZoneOffset.UTC), child.getDeadlineAt().toInstant(ZoneOffset.UTC),
                snapshot, order);
        return new Dispatch(started, child, parent, new CanvasActor(child.getUserId(), child.getActorTeamId()),
                scope.canvasId(), input);
    }
}
