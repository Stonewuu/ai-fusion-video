package com.stonewu.fusion.service.media.job;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.media.EpisodeAudioJobEntity;
import com.stonewu.fusion.mapper.media.EpisodeAudioJobMapper;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class EpisodeAudioReviewService {
    public record Step(String step, String providerTaskId, LocalDateTime submittedAt, boolean unresolved) {
    }

    public record View(String jobId, long revision, String status, boolean cancelRequested, boolean capacityHeld,
                       List<Step> steps, List<String> actions, ObjectNode lastReview) {
    }

    private final EpisodeAudioJobMapper audioJobs;
    private final MediaJobRepository jobs;
    private final EpisodeAudioJobs audio;
    private final CanvasAccessPolicy access;
    private final CanvasJson json;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View detail(String id, CanvasActor actor) {
        var job = jobs.get(id);
        if (!job.kind().equals("EPISODE_COMPOSE") || job.projectId() == null
                || !audioJobs.exists(Wrappers.<EpisodeAudioJobEntity>lambdaQuery().eq(EpisodeAudioJobEntity::getJobId, id))) {
            throw new MediaException(404, "EPISODE_AUDIO_NOT_FOUND", "此合成没有音频任务");
        }
        access.requireProject(job.projectId(), actor);
        var pending = audio.unresolved(id);
        var steps = List.of("music", "sfx").stream().flatMap(step -> jobs.submission(id, step).stream()
                .map(value -> new Step(step, value.providerId(), value.createdAt(), pending.contains(step)))).toList();
        String review = audioJobs.selectById(id).getReviewJson();
        var history = review == null ? null : json.read(review, ObjectNode.class);
        if (history != null) {
            history.remove("requestHash");
        }
        return new View(id, job.revision(), job.status(), job.cancelRequested(), audio.reserved(id), steps,
                List.of(), history);
    }
}
