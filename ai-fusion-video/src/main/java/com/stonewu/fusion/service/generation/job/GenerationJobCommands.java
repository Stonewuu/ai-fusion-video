package com.stonewu.fusion.service.generation.job;

import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import com.stonewu.fusion.service.generation.GenerationTasks;
import com.stonewu.fusion.mapper.generation.GenerationItemStateMapper;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.mapper.generation.RemoteGenerationJobMapper;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.mapper.generation.GenerationAttemptMapper;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.GenerationKind;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;

@Service
@RequiredArgsConstructor
public class GenerationJobCommands {
    private final GenerationTasks tasks;
    private final GenerationItemStateMapper itemStates;
    private final GenerationSubmissionMapper submissions;
    private final RemoteGenerationJobMapper remoteJobs;
    private final ProjectMapper projects;
    private final GenerationAttemptMapper attemptRows;
    private final GenerationJobRepository jobs;
    private final GenerationJobLifecycle lifecycle;
    private final GenerationAttemptRepository attempts;
    private final GenerationExecutionContexts access;
    private final GenerationOutbox outbox;
    private final GenerationReviewPolicy policy;

    @CacheEvict(value = {"imageTask", "videoTask", "imageItems", "videoItems"}, allEntries = true)
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public boolean cancel(GenerationKind kind, long taskId, CanvasActor actor) {
        var submission = jobs.byTask(kind, taskId).orElseThrow(() ->
                new GenerationJobException(409, "GENERATION_LEGACY_RECORD",
                        "此历史任务缺少可恢复执行记录，不能推断远端已取消"));
        control(submission, actor);
        submission = jobs.submission(submission.id());
        if (submission.cancelRequested() || jobs.jobs(submission.id()).stream()
                .allMatch(RemoteGenerationJob::terminal)) {
            return false;
        }
        submissions.update(new LambdaUpdateWrapper<GenerationSubmissionEntity>()
                .eq(GenerationSubmissionEntity::getId, submission.id())
                .set(GenerationSubmissionEntity::getCancelRequested, true));
        remoteJobs.requestCancellation(submission.id());
        for (var job : jobs.jobs(submission.id())) {
            if (job.phase().equals("NEW") && !job.state().equals("RUNNING") && !job.terminal()) {
                remoteJobs.cancelQueued(job.id());
                for (long itemId : job.itemIds()) {
                    itemStates.failPending(kind, itemId, "已取消排队");
                }
            }
        }
        lifecycle.aggregate(submission.id());
        return true;
    }

    @CacheEvict(value = {"imageTask", "videoTask", "imageItems", "videoItems"}, allEntries = true)
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void review(String jobId, Review decision, CanvasActor actor) {
        var before = jobs.get(jobId);
        var submission = jobs.submission(before.submissionId());
        control(submission, actor);
        remoteJobs.lock(jobId);
        var job = jobs.get(jobId);
        if (decision.action() == null) {
            throw GenerationJobException.invalid("请选择核实操作");
        }
        if (job.revision() != decision.expectedRevision()) {
            throw new GenerationJobException(409, "GENERATION_VERSION_CONFLICT", "任务状态已变化，请重新核对");
        }
        if (!job.requiresReview() && !job.state().equals("FAILED")) {
            throw GenerationJobException.invalid("当前任务不处于可核实或恢复的状态");
        }
        if (decision.note() == null || decision.note().isBlank() || decision.note().length() > 2000) {
            throw GenerationJobException.invalid("请填写核实依据（不超过 2000 字符）");
        }
        attemptRows.lockHistory(jobId);
        var history = attempts.forJob(jobId);
        if (!policy.allowed(job, history).contains(decision.action())) {
            throw GenerationJobException.invalid("此核实操作已不适用，请刷新任务状态");
        }
        switch (decision.action()) {
            case RESTORE_ACCEPTED_RECEIPT -> {
                var receipt = policy.lateReceipts(job, history).getFirst();
                resolveAttempt(job, decision, actor);
                remoteJobs.restoreReceipt(receipt.id(), jobId);
            }
            case RECHECK -> {
                if (job.activeAttemptId() == null) {
                    throw GenerationJobException.invalid("此执行没有已提交的请求");
                }
                var attempt = attempts.get(job.activeAttemptId());
                if (!attempt.acknowledged() && attempt.providerTaskId() == null) {
                    throw GenerationJobException.invalid("供应商未返回可查询标识，请先在供应商侧核实受理情况");
                }
                remoteJobs.recheck(jobId);
                for (long itemId : job.itemIds()) {
                    itemStates.resetFailed(submission.kind(), itemId);
                }
            }
            case RETRY_INGEST -> {
                if (job.result() == null || job.result()
                        .state() != RemoteGenerationResult.State.SUCCEEDED || job.cancelRequested()) {
                    throw GenerationJobException.invalid("没有可恢复接收的成功结果");
                }
                remoteJobs.retryIngest(jobId);
                for (long itemId : job.itemIds()) {
                    itemStates.resetFailed(submission.kind(), itemId);
                }
            }
            case CONFIRM_NOT_ACCEPTED_AND_RETRY -> {
                if (job.cancelRequested()) {
                    throw GenerationJobException.invalid("已请求取消的执行不能重新提交，请新建生成任务");
                }
                if (job.activeAttemptId() == null) {
                    throw GenerationJobException.invalid("没有需要核实的外部提交");
                }
                attemptRows.lock(job.activeAttemptId());
                var attempt = attempts.get(job.activeAttemptId());
                if (attempt.acknowledged()) {
                    throw GenerationJobException.invalid("供应商已确认受理，不能将其改为未受理后重发");
                }
                resolveAttempt(job, decision, actor);
                remoteJobs.retrySubmission(jobId);
                for (long itemId : job.itemIds()) {
                    itemStates.resetFailed(submission.kind(), itemId);
                }
            }
            case CONFIRM_PROVIDER_FINISHED -> {
                if (!job.requiresReview()) {
                    throw GenerationJobException.invalid("只有状态待核实的任务需要确认远端结束");
                }
                resolveAttempt(job, decision, actor);
                String reason = "已核实远端结束，未取得可用结果";
                String pendingPrefix = "请求结果未确认，不能自动重发：";
                if (job.errorMessage() != null && job.errorMessage().startsWith(pendingPrefix)) {
                    reason = job.errorMessage().substring(pendingPrefix.length());
                }
                remoteJobs.confirmFinished(job.cancelRequested() ? "CANCELLED" : "FAILED", reason, jobId);
                for (long itemId : job.itemIds()) {
                    itemStates.failPending(submission.kind(), itemId, reason);
                }
            }
        }
        lifecycle.aggregate(submission.id());
        long revision = jobs.submission(submission.id()).revision();
        outbox.enqueue("generation-review:" + jobId + ":" + revision, "GENERATION_CHANGED", submission.id(), Map.of(
                "submissionId", submission.id(), "revision", revision, "jobId", jobId, "action",
                decision.action().name(),
                "actorUserId", actor.userId(), "note", decision.note()));
    }

    private void resolveAttempt(RemoteGenerationJob job, Review decision, CanvasActor actor) {
        if (job.activeAttemptId() == null) {
            return;
        }
        attemptRows.resolve(decision.action().name(), decision.note(), actor.userId(), job.activeAttemptId());
    }

    private void control(GenerationSubmission submission, CanvasActor actor) {
        access.requireRead(submission.kind(), submission.taskId(), actor);
        if (submission.userId() != actor.userId()) {
            throw new GenerationJobException(403, "GENERATION_CONTROL_DENIED", "只能操作自己发起的生成任务");
        }
        if (submission.projectId() != null) {
            projects.lock(submission.projectId());
        }
        tasks.lock(submission.kind(), submission.taskId());
        submissions.lock(submission.id());
    }

    public enum Action {RECHECK, RETRY_INGEST, RESTORE_ACCEPTED_RECEIPT, CONFIRM_NOT_ACCEPTED_AND_RETRY, CONFIRM_PROVIDER_FINISHED}

    public record Review(long expectedRevision, Action action, String note) {
    }
}
