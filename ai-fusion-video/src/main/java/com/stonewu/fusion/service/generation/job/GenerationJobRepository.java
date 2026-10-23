package com.stonewu.fusion.service.generation.job;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.support.SFunction;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import com.stonewu.fusion.entity.generation.RemoteGenerationJobEntity;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.mapper.generation.RemoteGenerationJobMapper;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;
import com.stonewu.fusion.mapper.ai.AiModelMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
@RequiredArgsConstructor
public class GenerationJobRepository {
    private final GenerationSubmissionMapper submissions;
    private final RemoteGenerationJobMapper jobMapper;
    private final AiModelMapper models;
    private final ExecutionCoordinationMapper coordination;
    private final CanvasJson json;
    @Value("${app.generation.max-user-remote-jobs:32}")
    private int maxUserJobs;
    @Value("${app.generation.max-project-remote-jobs:64}")
    private int maxProjectJobs;

    public GenerationSubmission submission(String id) {
        return Optional.ofNullable(submissions.selectById(id)).map(this::readSubmission)
                .orElseThrow(() -> new GenerationJobException(404, "GENERATION_NOT_FOUND", "生成受理记录不存在"));
    }

    public Optional<GenerationSubmission> byTask(GenerationKind kind, long taskId) {
        return Optional.ofNullable(submissions.selectOne(new LambdaQueryWrapper<GenerationSubmissionEntity>()
                        .eq(GenerationSubmissionEntity::getKind, kind.name())
                        .eq(GenerationSubmissionEntity::getTaskId, taskId)))
                .map(this::readSubmission);
    }

    public RemoteGenerationJob get(String id) {
        return Optional.ofNullable(jobMapper.selectById(id)).map(this::readJob)
                .orElseThrow(() -> new GenerationJobException(404, "GENERATION_JOB_NOT_FOUND", "远端生成记录不存在"));
    }

    public List<RemoteGenerationJob> jobs(String submissionId) {
        return jobMapper.selectList(new LambdaQueryWrapper<RemoteGenerationJobEntity>()
                        .eq(RemoteGenerationJobEntity::getSubmissionId, submissionId)
                        .orderByAsc(RemoteGenerationJobEntity::getItemIndex))
                .stream().map(this::readJob).toList();
    }

    /**
     * 额度检查和领取只持有短事务锁；远端 HTTP 不在事务内执行。
     */
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Optional<RemoteGenerationJob> claim(String owner) {
        coordination.lockGenerationDispatch();
        // 排序只读取领取和额度字段；大体积输入、图片结果在领取后按主键加载。
        var candidates = jobMapper.selectClaimCandidates();
        for (var job : candidates) {
            boolean needsPermit = !job.getPermitHeld() && !job.getCancelRequested() && !job.getPhase().equals("INGESTING");
            if (needsPermit && !hasCapacity(job)) {
                jobMapper.deferForCapacity(job.getId());
                continue;
            }
            jobMapper.claim(owner, job.getPermitHeld() || needsPermit, job.getId());
            return Optional.of(get(job.getId()));
        }
        return Optional.empty();
    }

    private boolean hasCapacity(RemoteGenerationJobEntity job) {
        var model = models.selectOne(new LambdaQueryWrapper<AiModel>()
                .select(AiModel::getMaxConcurrency).eq(AiModel::getId, job.getModelId()));
        // 配置失效的任务仍须被领取以记录明确错误，不能永久滞留队列。
        if (model == null || model.getMaxConcurrency() == null || model.getMaxConcurrency() < 1) {
            return true;
        }
        if (held(RemoteGenerationJobEntity::getModelId, job.getModelId()) >= model.getMaxConcurrency()) {
            return false;
        }
        if (held(RemoteGenerationJobEntity::getActorUserId, job.getActorUserId()) >= Math.max(1, maxUserJobs)) {
            return false;
        }
        return job.getProjectId() == null || held(RemoteGenerationJobEntity::getProjectId, job.getProjectId()) < Math.max(1,
                maxProjectJobs);
    }

    private long held(SFunction<RemoteGenerationJobEntity, ?> column, long value) {
        return jobMapper.selectCount(new LambdaQueryWrapper<RemoteGenerationJobEntity>()
                .eq(column, value).eq(RemoteGenerationJobEntity::getPermitHeld, true));
    }

    public boolean heartbeat(RemoteGenerationJob.Claim claim) {
        return jobMapper.heartbeat(claim.id(), claim.owner(), claim.epoch()) == 1;
    }

    /**
     * 所有状态提交按受理记录、子执行的顺序加锁。
     */
    @Transactional(propagation = Propagation.MANDATORY)
    public RemoteGenerationJob leased(RemoteGenerationJob.Claim claim, boolean allowCancel) {
        var before = get(claim.id());
        submissions.lock(before.submissionId());
        var row = jobMapper.selectLeasedForUpdate(claim.id(), claim.owner(), claim.epoch());
        if (row == null || (!allowCancel && row.getCancelRequested())) {
            throw GenerationJobException.fenced();
        }
        return readJob(row);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public RemoteGenerationJob check(RemoteGenerationJob.Claim claim) {
        return leased(claim, false);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void waitForNextPoll(RemoteGenerationJob.Claim claim, int seconds, String phase,
                                String errorCode, String errorMessage) {
        var job = leased(claim, true);
        jobMapper.schedulePoll(phase, Math.max(1, Math.min(60, seconds)),
                errorCode == null ? 0 : job.consecutiveErrors() + 1,
                errorCode, errorMessage, job.id());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void remoteResult(RemoteGenerationJob.Claim claim, RemoteGenerationResult result) {
        var job = leased(claim, true);
        if (result.state() == RemoteGenerationResult.State.PENDING) {
            throw GenerationJobException.invalid("等待结果不能写入完成检查点");
        }
        jobMapper.receiveResult(json.write(result), job.id());
    }

    public LocalDateTime now() {
        return coordination.now();
    }

    private GenerationSubmission readSubmission(GenerationSubmissionEntity row) {
        var model = json.read(row.getModelSnapshotJson(), AiModel.class);
        model.setGenerationConfigSnapshot(true);
        return new GenerationSubmission(row.getId(), row.getRequestId(),
                GenerationKind.valueOf(row.getKind()),
                row.getTaskId(), row.getPublicTaskId(), row.getActorUserId(),
                row.getActorTeamId(),
                row.getProjectId(), row.getCanvasNodeRunId(),
                json.read(row.getContextJson(), ResourceCommandContext.class),
                model, row.getApiConfigId(),
                row.getApiConfigHash(),
                row.getState(), row.getCancelRequested(), row.getRevision(),
                row.getCreatedAt());
    }

    private RemoteGenerationJob readJob(RemoteGenerationJobEntity row) {
        var lease = row.getLeaseUntil();
        String result = row.getResultJson();
        return new RemoteGenerationJob(row.getId(), row.getSubmissionId(), row.getItemId(),
                row.getItemIndex(), row.getModelId(), row.getActorUserId(),
                row.getProjectId(),
                row.getState(), row.getPhase(), row.getProviderTaskId(),
                row.getActiveAttemptId(),
                json.read(row.getInputJson(), ObjectNode.class),
                result == null ? null : json.read(result, RemoteGenerationResult.class),
                row.getCancelRequested(), row.getCancelSent(), row.getPermitHeld(),
                row.getOwnerInstanceId(),
                row.getOwnerEpoch(), lease,
                row.getDeadlineAt(),
                row.getPollAttempts(), row.getConsecutiveErrors(), row.getErrorCode(),
                row.getErrorMessage(), row.getRevision());
    }
}
