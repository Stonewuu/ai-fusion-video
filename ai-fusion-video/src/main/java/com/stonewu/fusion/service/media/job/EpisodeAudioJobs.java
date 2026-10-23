package com.stonewu.fusion.service.media.job;

import com.stonewu.fusion.mapper.media.EpisodeAudioJobMapper;
import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.entity.media.EpisodeAudioJobEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 只管理原分集音频调用的证据和额度；不代替原身份发布媒体或修改分镜。
 */
@Component
@RequiredArgsConstructor
public class EpisodeAudioJobs {
    private final EpisodeAudioJobMapper audioJobs;
    private final MediaJobMapper mediaJobs;
    private final CanvasJson json;
    private final MediaJobRepository jobs;
    private final MediaJobNotifications notifications;

    public boolean reserved(String id) {
        return audioJobs.exists(Wrappers.<EpisodeAudioJobEntity>lambdaQuery()
                .eq(EpisodeAudioJobEntity::getJobId, id).eq(EpisodeAudioJobEntity::getPermitHeld, true));
    }

    public List<String> unresolved(String id) {
        return audioJobs.unresolved(id);
    }

    public boolean hasSuccessfulResult(String id) {
        return audioJobs.successfulResult(id) != null;
    }

    public ObjectNode result(String id, String step) {
        String field = field(step);
        String value = field.equals("music") ? audioJobs.selectById(id).getMusicResultJson() : audioJobs.selectById(id)
                .getSfxResultJson();
        return value == null ? null : json.read(value, ObjectNode.class);
    }

    public LocalDateTime pollDeadline(String id, String step, int timeoutSeconds) {
        var started = jobs.submission(id, step).orElseThrow(MediaJobFence::lost).createdAt();
        String review = audioJobs.selectById(id).getReviewJson();
        if (review != null) {
            var value = json.read(review, ObjectNode.class);
            if (List.of("RECHECK", "RECORD_RECEIPT").contains(value.path("action").asText())) {
                var reviewed = LocalDateTime.parse(value.path("recordedAt").asText());
                if (reviewed.isAfter(started)) {
                    started = reviewed;
                }
            }
        }
        return started.plusSeconds(timeoutSeconds);
    }

    /**
     * 原查询的迟到终态也可入账和释放额度，但不能越过新租约写回业务。
     */
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void received(String id, String step, String attempt, ObjectNode result) {
        saveResult(id, step, attempt, result, false);
    }

    /**
     * 提交响应明确失败，无远端任务编号；保留原尝试和失败响应后释放额度。
     */
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void rejected(String id, String step, String attempt, GenerationProviderFailure failure) {
        saveResult(id, step, attempt, json.object(Map.of("status", "failed", "error", failure.getMessage())), true);
    }

    private void saveResult(String id, String step, String attempt, ObjectNode result, boolean rejected) {
        String field = field(step);
        if (!List.of("succeeded", "failed").contains(result.path("status").asText())
                || json.write(result).getBytes(StandardCharsets.UTF_8).length > 4 * 1024 * 1024) {
            throw MediaException.invalid("音频终态响应无效或过大");
        }
        mediaJobs.lock(id);
        var submission = jobs.submission(id, step).orElseThrow(MediaJobFence::lost);
        if (!submission.attempt().equals(attempt) || (!rejected && submission.providerId() == null)) {
            throw MediaJobFence.lost();
        }
        String hash = json.hash(result);
        String previous = field.equals("music") ? audioJobs.lock(id).getMusicResultHash() : audioJobs.lock(id)
                .getSfxResultHash();
        if (previous != null) {
            if (!previous.equals(hash)) {
                throw new MediaException(409, "MEDIA_AUDIO_RESULT_CONFLICT", "原音频终态响应不能改写");
            }
            return;
        }
        audioJobs.update(Wrappers.<EpisodeAudioJobEntity>lambdaUpdate().eq(EpisodeAudioJobEntity::getJobId, id)
                .set(field.equals(
                                "music") ? EpisodeAudioJobEntity::getMusicResultJson : EpisodeAudioJobEntity::getSfxResultJson,
                        json.write(result))
                .set(field.equals(
                                "music") ? EpisodeAudioJobEntity::getMusicResultHash : EpisodeAudioJobEntity::getSfxResultHash,
                        hash)
                .set(EpisodeAudioJobEntity::getConsecutiveErrors, 0));
        audioJobs.update(Wrappers.<EpisodeAudioJobEntity>lambdaUpdate().eq(EpisodeAudioJobEntity::getJobId, id)
                .set(EpisodeAudioJobEntity::getPermitHeld, !unresolved(id).isEmpty()));
        mediaJobs.advanceRevision(id);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void waitNext(MediaJob.Claim claim, String phase, int seconds, boolean failedQuery) {
        var job = jobs.observe(claim);
        audioJobs.pollResult(job.id(), failedQuery);
        int errors = audioJobs.selectById(job.id()).getConsecutiveErrors();
        boolean review = errors >= 8;
        mediaJobs.waitForAudio(review ? "NEEDS_REVIEW" : "WAITING", review ? "音频任务等待核实" : phase,
                review ? "连续查询原音频任务失败，请核实供应商是否已结束" : null,
                Math.max(1, Math.min(60, failedQuery ? 3 + errors * 3 : seconds)), job.id());
        notifications.progress(jobs.get(job.id()));
    }

    private static String field(String step) {
        if (!step.equals("music") && !step.equals("sfx")) {
            throw MediaException.invalid("无效的分集音频步骤");
        }
        return step;
    }
}
