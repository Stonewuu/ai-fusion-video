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
            throw GenerationJobException.invalid("请选择处理操作");
        }
        if (job.revision() != decision.expectedRevision()) {
            throw new GenerationJobException(409, "GENERATION_VERSION_CONFLICT", "任务状态已变化，请重新核对");
        }
        if (!job.requiresReview() && !job.state().equals("FAILED")) {
            throw GenerationJobException.invalid("当前任务不能重新保存结果");
        }
        if (decision.note() == null || decision.note().isBlank() || decision.note().length() > 2000) {
            throw GenerationJobException.invalid("处理说明不能为空且不能超过 2000 字符");
        }
        attemptRows.lockHistory(jobId);
        var history = attempts.forJob(jobId);
        if (!policy.allowed(job, history).contains(decision.action())) {
            throw GenerationJobException.invalid("当前无法重新保存结果，请刷新任务状态");
        }
        remoteJobs.retryIngest(jobId);
        for (long itemId : job.itemIds()) {
            itemStates.resetFailed(submission.kind(), itemId);
        }
        lifecycle.aggregate(submission.id());
        long revision = jobs.submission(submission.id()).revision();
        outbox.enqueue("generation-review:" + jobId + ":" + revision, "GENERATION_CHANGED", submission.id(), Map.of(
                "submissionId", submission.id(), "revision", revision, "jobId", jobId, "action",
                decision.action().name(),
                "actorUserId", actor.userId(), "note", decision.note()));
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

    public enum Action {RETRY_INGEST}

    public record Review(long expectedRevision, Action action, String note) {
    }
}
