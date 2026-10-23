package com.stonewu.fusion.service.canvas.agent;

import com.stonewu.fusion.mapper.ai.AgentModelCallUsageMapper;
import com.stonewu.fusion.entity.ai.AgentModelCallUsage;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.run.model.ResumedAgentRun;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Objects;

@Repository
@RequiredArgsConstructor
public class CanvasAgentWaitRepository {
    public record Dispatch(ResumedAgentRun run, String resultJson, long sequence) {
    }

    private final AgentModelCallUsageMapper usage;
    private final AgentRunMapper runs;
    private final AgentScopeV2Properties properties;

    public List<String> waiting() {
        return runs.waitingForCanvas();
    }

    public List<String> dispatching(String instance) {
        return runs.dispatchingCanvasResume(instance);
    }

    public AgentRun deferCheck(String id) {
        if (runs.deferCanvasCheck(id) != 1) {
            return null;
        }
        return runs.selectOne(new LambdaQueryWrapper<AgentRun>().eq(AgentRun::getRunId, id));
    }

    @Transactional
    public Dispatch claim(String id, String instance) {
        var run = runs.selectByRunIdForUpdate(id);
        LocalDateTime now = runs.selectDatabaseNow();
        if (run == null || !"RUNNING".equals(run.getStatus()) || run.getExternalResumeJson() == null
                || run.getCancelRequestedAt() != null || !run.getDeadlineAt().isAfter(now)) {
            return null;
        }
        boolean expired = run.getLeaseUntil() == null || !run.getLeaseUntil().isAfter(now);
        if (!expired && !Objects.equals(instance, run.getOwnerInstanceId())) {
            return null;
        }
        long epoch = run.getOwnerEpoch();
        LocalDateTime lease = run.getLeaseUntil();
        if (expired) {
            boolean untouched = Objects.equals(run.getNextSequence(), run.getExternalResumeSequence() + 1);
            long calls = usage.selectCount(Wrappers.<AgentModelCallUsage>lambdaQuery()
                    .eq(AgentModelCallUsage::getRunId, id)
                    .ge(AgentModelCallUsage::getStartedAt, run.getExternalResumeAt()));
            if (!untouched || calls != 0) {
                // 已开始执行时不自动重放模型请求；交还已有的 owner 丢失与显式继续流程。
                clear(id, epoch, run.getExternalResumeSequence());
                return null;
            }
            epoch++;
            lease = now.plus(properties.getExecution().getOwnerLease());
            if (lease.isAfter(run.getDeadlineAt())) {
                lease = run.getDeadlineAt();
            }
            runs.update(Wrappers.<AgentRun>lambdaUpdate().eq(AgentRun::getRunId, id)
                    .eq(AgentRun::getStatus, "RUNNING").eq(AgentRun::getOwnerEpoch, run.getOwnerEpoch())
                    .set(AgentRun::getOwnerInstanceId, instance).set(AgentRun::getOwnerEpoch, epoch)
                    .set(AgentRun::getLeaseUntil, lease).set(AgentRun::getHeartbeatAt, now)
                    .set(AgentRun::getUpdateTime, now));
        }
        return new Dispatch(new ResumedAgentRun(id, run.getConversationId(), run.getAgentStateSessionId(),
                run.getKernelFingerprint(), run.getAgentDefinitionSnapshotJson(), run.getPausedThroughSequence(),
                instance, epoch,
                lease.toInstant(ZoneOffset.UTC), run.getDeadlineAt().toInstant(ZoneOffset.UTC)),
                run.getExternalResumeJson(), run.getExternalResumeSequence());
    }

    public void clear(String id, long epoch, long sequence) {
        runs.clearCanvasResume(id, epoch, sequence);
    }
}
