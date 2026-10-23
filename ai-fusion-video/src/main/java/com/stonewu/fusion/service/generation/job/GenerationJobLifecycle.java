package com.stonewu.fusion.service.generation.job;

import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import com.stonewu.fusion.entity.generation.RemoteGenerationJobEntity;
import com.stonewu.fusion.service.generation.GenerationTasks;
import com.stonewu.fusion.mapper.generation.GenerationItemStateMapper;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.mapper.generation.RemoteGenerationJobMapper;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.mapper.generation.ImageItemMapper;
import com.stonewu.fusion.mapper.generation.VideoItemMapper;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.media.MediaReference;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * 结果发布与聚合终态同事务提交；传输、下载及 FFmpeg 都在事务外。
 */
@Service
@RequiredArgsConstructor
public class GenerationJobLifecycle {
    private final GenerationTasks tasks;
    private final GenerationItemStateMapper itemStates;
    private final GenerationSubmissionMapper submissions;
    private final RemoteGenerationJobMapper remoteJobs;
    private final ProjectMapper projects;
    private final GenerationJobRepository jobs;
    private final GenerationOutbox outbox;
    private final GenerationExecutionContexts contexts;
    private final ImageGenerationService images;
    private final VideoGenerationService videos;
    private final ImageItemMapper imageItems;
    private final VideoItemMapper videoItems;

    @CacheEvict(value = {"imageTask", "videoTask"}, allEntries = true)
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public GenerationSubmission start(RemoteGenerationJob.Claim claim) {
        var submission = jobs.submission(jobs.get(claim.id()).submissionId());
        lockTask(submission);
        jobs.leased(claim, true);
        tasks.start(submission.kind(), submission.taskId());
        aggregate(submission.id());
        return jobs.submission(submission.id());
    }

    @CacheEvict(value = {"imageTask", "videoTask", "imageItems", "videoItems"}, allEntries = true)
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void succeed(RemoteGenerationJob.Claim claim, List<RemoteGenerationResult.Output> outputs) {
        var submission = jobs.submission(jobs.get(claim.id()).submissionId());
        contexts.running(submission.kind(), submission.taskId());
        var job = jobs.leased(claim, false);
        if (!job.phase().equals("INGESTING") || job.result() == null) {
            throw GenerationJobException.invalid("缺少已完成远端结果检查点");
        }
        if (outputs.isEmpty() || outputs.size() > job.itemIds().size()) {
            throw GenerationJobException.invalid("结果数量与固定条目集合不符");
        }
        for (int index = 0; index < outputs.size(); index++) {
            publish(submission, job, job.itemIds().get(index), outputs.get(index));
        }
        boolean partial = outputs.size() < job.itemIds().size();
        if (partial) {
            for (long missing : job.itemIds().subList(outputs.size(), job.itemIds().size())) {
                itemStates.failPending(submission.kind(), missing, "供应商返回的结果数量不足");
            }
        }
        terminalRow(job.id(), partial ? "PARTIAL_FAILED" : "SUCCEEDED", partial ? "PROVIDER_OUTPUTS_MISSING" : null,
                partial ? "供应商返回的结果数量不足，已保留实际完成的结果" : null);
        aggregate(submission.id());
    }

    private void publish(GenerationSubmission submission, RemoteGenerationJob job, long itemId,
                         RemoteGenerationResult.Output output) {
        if (!MediaReference.managed(output.media())) {
            throw GenerationJobException.invalid("输出尚未完成托管登记");
        }
        if (submission.kind() == GenerationKind.IMAGE) {
            ImageItem item = imageItems.selectById(itemId);
            if (item == null || !item.getTaskId().equals(submission.taskId())) {
                throw GenerationJobException.fenced();
            }
            item.setPlatformTaskId(job.providerTaskId());
            item.setImageUrl(output.media());
            item.setThumbnailUrl(output.cover());
            item.setStatus(1);
            item.setErrorMsg(null);
            images.updateItem(item);
        } else {
            VideoItem item = videoItems.selectById(itemId);
            if (item == null || !item.getTaskId().equals(submission.taskId())) {
                throw GenerationJobException.fenced();
            }
            item.setPlatformTaskId(job.providerTaskId());
            item.setVideoUrl(output.media());
            item.setCoverUrl(output.cover());
            item.setFirstFrameUrl(output.firstFrame());
            item.setLastFrameUrl(output.lastFrame());
            item.setDuration(output.durationSeconds());
            item.setStatus(1);
            item.setErrorMsg(null);
            videos.updateItem(item);
        }
    }

    @CacheEvict(value = {"imageTask", "videoTask", "imageItems", "videoItems"}, allEntries = true)
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void fail(RemoteGenerationJob.Claim claim, String state, String code, String message) {
        if (!List.of("FAILED", "CANCELLED", "UNKNOWN_SUBMISSION", "NEEDS_REVIEW").contains(state)) {
            throw GenerationJobException.invalid("无效的任务结束状态");
        }
        var submission = jobs.submission(jobs.get(claim.id()).submissionId());
        // 清理已失去业务权限的运行只写自身任务，不发布业务内容。
        lockTask(submission);
        var job = jobs.leased(claim, true);
        boolean review = state.equals("UNKNOWN_SUBMISSION") || state.equals("NEEDS_REVIEW");
        if (review) {
            remoteJobs.markReview(state, code, message, job.id());
        } else {
            terminalRow(job.id(), state, code, message);
            for (long itemId : job.itemIds()) {
                itemStates.failTaskItem(submission.kind(), itemId, submission.taskId(), message);
            }
        }
        aggregate(submission.id());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void cancellationSent(RemoteGenerationJob.Claim claim) {
        var job = jobs.leased(claim, true);
        remoteJobs.update(new LambdaUpdateWrapper<RemoteGenerationJobEntity>()
                .eq(RemoteGenerationJobEntity::getId, job.id())
                .set(RemoteGenerationJobEntity::getCancelSent, true)
                .setIncrBy(RemoteGenerationJobEntity::getRevision, 1));
        aggregate(job.submissionId());
    }

    private void terminalRow(String id, String state, String code, String message) {
        remoteJobs.finish(state, code, message, id);
    }

    /**
     * 调用方已锁任务行，之后始终按父受理、子执行加锁。
     */
    @Transactional(propagation = Propagation.MANDATORY)
    public void aggregate(String id) {
        submissions.lock(id);
        var submission = jobs.submission(id);
        var children = jobs.jobs(id);
        long succeeded = tasks.succeeded(submission.kind(), submission.taskId());
        int expected = children.stream().mapToInt(job -> job.itemIds().size()).sum();
        boolean complete = children.stream().allMatch(RemoteGenerationJob::terminal);
        String state;
        int taskStatus;
        if (complete) {
            state = succeeded == expected ? "SUCCEEDED"
                    : submission.cancelRequested() ? "CANCELLED" : succeeded > 0 ? "PARTIAL_FAILED" : "FAILED";
            taskStatus = state.equals("SUCCEEDED") ? 2 : state.equals("CANCELLED") ? 4 : state.equals(
                    "PARTIAL_FAILED") ? 6 : 3;
        } else {
            state = submission.cancelRequested() ? "CANCEL_REQUESTED"
                    : children.stream().anyMatch(job -> job.state().equals("UNKNOWN_SUBMISSION")) ? "UNKNOWN_SUBMISSION"
                    : children.stream().anyMatch(RemoteGenerationJob::requiresReview) ? "NEEDS_REVIEW" : "RUNNING";
            taskStatus = 1;
        }
        String error = children.stream().filter(job -> job.errorMessage() != null)
                .map(RemoteGenerationJob::errorMessage).findFirst().orElse(null);
        tasks.aggregate(submission.kind(), submission.taskId(), taskStatus, succeeded, error);
        submissions.aggregate(state, id);
        long revision = submission.revision() + 1;
        outbox.enqueue("generation:" + id + ":" + revision, "GENERATION_CHANGED", id,
                GenerationOutbox.state(id, revision, state));
    }

    private void lockTask(GenerationSubmission submission) {
        if (submission.projectId() != null) {
            projects.lock(submission.projectId());
        }
        tasks.lock(submission.kind(), submission.taskId());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void requestStop(RemoteGenerationJob.Claim claim) {
        var submission = jobs.submission(jobs.get(claim.id()).submissionId());
        lockTask(submission);
        jobs.leased(claim, true);
        submissions.update(new LambdaUpdateWrapper<GenerationSubmissionEntity>()
                .eq(GenerationSubmissionEntity::getId, submission.id())
                .set(GenerationSubmissionEntity::getCancelRequested, true));
        remoteJobs.update(new LambdaUpdateWrapper<RemoteGenerationJobEntity>()
                .eq(RemoteGenerationJobEntity::getSubmissionId, submission.id())
                .notIn(RemoteGenerationJobEntity::getState, "SUCCEEDED", "PARTIAL_FAILED", "FAILED", "CANCELLED")
                .set(RemoteGenerationJobEntity::getCancelRequested, true));
        aggregate(submission.id());
    }
}
