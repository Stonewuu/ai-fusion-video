package com.stonewu.fusion.service.media.job;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.stonewu.fusion.entity.media.MediaJobSubmissionEntity;

import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.media.MediaJobSubmissionMapper;
import com.stonewu.fusion.entity.media.MediaJobEntity;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Optional;
import java.util.UUID;

@Repository
@RequiredArgsConstructor
public class MediaJobRepository {
    private final MediaJobMapper mediaJobs;
    private final MediaJobSubmissionMapper submissions;
    private final ExecutionCoordinationMapper coordination;
    private final CanvasJson json;
    private final MediaJobFence fence;
    private final MediaJobNotifications notifications;

    public MediaJob get(String id) {
        var row = mediaJobs.selectById(id);
        if (row == null) {
            throw new MediaException(404, "MEDIA_JOB_NOT_FOUND", "媒体任务不存在");
        }
        return read(row);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Optional<MediaJob> claim(String owner) {
        coordination.lockGenerationDispatch();
        var rows = mediaJobs.nextCompose();
        if (rows.isEmpty()) {
            return Optional.empty();
        }
        String id = rows.getFirst().getId();
        mediaJobs.claimCompose(owner, id);
        mediaJobs.acquireEpisodeAudioPermit(id);
        return Optional.of(get(id));
    }

    public boolean heartbeat(MediaJob.Claim claim) {
        return mediaJobs.heartbeat(claim.id(), claim.owner(), claim.epoch()) == 1;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public MediaJob check(MediaJob.Claim claim) {
        MediaJob job = get(claim.id());
        fence.require(claim, job.context(), false);
        return get(claim.id());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public MediaJob observe(MediaJob.Claim claim) {
        MediaJob job = get(claim.id());
        fence.require(claim, job.context(), true);
        return get(claim.id());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public MediaJob checkpoint(MediaJob.Claim claim, ObjectNode checkpoint, String phase) {
        MediaJob job = get(claim.id());
        fence.require(claim, job.context(), false);
        mediaJobs.checkpoint(json.write(checkpoint), phase, claim.id());
        var current = get(claim.id());
        notifications.progress(current);
        return current;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void waitForProvider(MediaJob.Claim claim, int seconds) {
        MediaJob job = get(claim.id());
        fence.require(claim, job.context(), false);
        mediaJobs.waitForProvider(Math.max(1, Math.min(60, seconds)), claim.id());
    }

    /**
     * 付费调用前提交意图。已有意图而无回执时由调用者停止，不能重复提交。
     */
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Submission prepareSubmission(MediaJob.Claim claim, String step) {
        MediaJob job = get(claim.id());
        fence.require(claim, job.context(), false);
        var previous = submission(claim.id(), step);
        if (previous.isPresent()) {
            return previous.get();
        }
        if (job.kind().equals("EPISODE_COMPOSE")) {
            if (!step.equals("music") && !step.equals("sfx")) {
                throw MediaException.invalid("无效的分集音频步骤");
            }
            if (mediaJobs.lockEpisodeAudioPermit(job.id()).isEmpty()) {
                throw new MediaException(409, "MEDIA_AUDIO_CAPACITY", "等待音频并发额度");
            }
        }
        String attempt = UUID.randomUUID().toString();
        submissions.prepare(claim.id(), step, attempt);
        return new Submission(attempt, null, true, submission(claim.id(), step).orElseThrow().createdAt());
    }

    public Optional<Submission> submission(String id, String step) {
        return Optional.ofNullable(submissions.selectOne(Wrappers.<MediaJobSubmissionEntity>lambdaQuery()
                        .eq(MediaJobSubmissionEntity::getJobId, id).eq(MediaJobSubmissionEntity::getStep, step)))
                .map(row -> new Submission(row.getAttemptId(), row.getProviderTaskId(), false, row.getCreatedAt()));
    }

    /**
     * 已发出的请求即使失去租约，也只补记不可变回执；它没有修改业务内容或继续执行的权利。
     */
    public void acknowledge(String attempt, String providerId) {
        if (providerId == null || providerId.isBlank() || providerId.length() > 256) {
            throw MediaException.invalid("供应商任务标识无效");
        }
        int changed = submissions.acknowledge(providerId, attempt);
        if (changed != 1) {
            throw new MediaException(409, "MEDIA_SUBMISSION_CONFLICT", "供应商回执与提交意图不符");
        }
    }

    public LocalDateTime now() {
        return coordination.now();
    }

    public record Submission(String attempt, String providerId, boolean created, LocalDateTime createdAt) {
    }

    private MediaJob read(MediaJobEntity row) {
        return new MediaJob(row.getId(), row.getKind(), row.getProjectId(),
                row.getTargetId(),
                row.getTargetVersion(),
                json.read(row.getContextJson(), ResourceCommandContext.class),
                json.read(row.getInputJson(), ObjectNode.class),
                json.read(row.getCheckpointJson(), ObjectNode.class),
                row.getStatus(), row.getPhase(), row.getErrorMsg(),
                row.getOwnerInstanceId(),
                row.getOwnerEpoch(), row.getCancelRequested(), row.getRevision(),
                row.getDeadlineAt(), row.getCreatedAt());
    }
}
