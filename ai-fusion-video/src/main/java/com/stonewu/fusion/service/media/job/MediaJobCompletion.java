package com.stonewu.fusion.service.media.job;

import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.media.EpisodeAudioJobMapper;
import com.stonewu.fusion.entity.media.EpisodeAudioJobEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.model.ComposeWritebackCommand;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Set;

@Component
@RequiredArgsConstructor
public class MediaJobCompletion {
    private final MediaJobMapper mediaJobs;
    private final EpisodeAudioJobMapper audioJobs;
    private final MediaJobRepository jobs;
    private final MediaJobFence fence;
    private final ResourceCommandService commands;
    private final MediaJobNotifications notifications;
    private final EpisodeAudioJobs audio;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void complete(MediaJob.Claim claim) {
        var job = jobs.get(claim.id());
        fence.require(claim, job.context(), false);
        commands.execute(new ComposeWritebackCommand("compose:" + job.id() + ":complete", job.targetId(), job.id(),
                ComposeWritebackCommand.Stage.COMPLETE, claim), job.context());
        terminal(claim, "SUCCEEDED", null);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public boolean recordFailure(MediaJob.Claim claim, String error) {
        return mediaJobs.recordFailure(error, claim.id(), claim.owner(), claim.epoch()) == 1;
    }

    /**
     * 独立的业务事务失败不妨碍任务自身结束；撤权后绝不冒用系统身份写回。
     */
    public void cleanBusinessStatus(MediaJob.Claim claim) {
        var job = jobs.get(claim.id());
        try {
            commands.execute(new ComposeWritebackCommand("compose:" + job.id() + ":end", job.targetId(), job.id(),
                    ComposeWritebackCommand.Stage.END, claim), job.context());
        } catch (CanvasException | MediaException failure) {
            if (failure.getCode() != 403 && failure.getCode() != 404 && failure.getCode() != 409) {
                throw failure;
            }
        }
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void terminal(MediaJob.Claim claim, String status, String error) {
        if (!Set.of("SUCCEEDED", "FAILED", "CANCELLED", "CONFLICT", "UNKNOWN_SUBMISSION").contains(status)) {
            throw MediaException.invalid("无效的媒体终态");
        }
        mediaJobs.lock(claim.id());
        boolean unresolved = !audio.unresolved(claim.id()).isEmpty();
        if (unresolved && !status.equals("UNKNOWN_SUBMISSION")) {
            throw new MediaException(409, "MEDIA_AUDIO_UNRESOLVED", "原音频请求尚未确认结束，必须保留远端结果未知");
        }
        int changed = mediaJobs.terminal(status, status.equals("SUCCEEDED") ? "合成完成" : status.equals(
                "UNKNOWN_SUBMISSION") ? "远端结果未知" : "处理已结束", error, claim.id(), claim.owner(), claim.epoch());
        if (changed == 1) {
            audioJobs.update(
                    Wrappers.<EpisodeAudioJobEntity>lambdaUpdate().eq(EpisodeAudioJobEntity::getJobId, claim.id())
                            .set(EpisodeAudioJobEntity::getPermitHeld, false));
            notifications.finished(jobs.get(claim.id()));
        }
    }
}
