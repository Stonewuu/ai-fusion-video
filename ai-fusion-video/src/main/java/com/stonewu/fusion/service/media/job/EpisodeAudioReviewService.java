package com.stonewu.fusion.service.media.job;

import com.stonewu.fusion.entity.media.MediaJobSubmissionEntity;

import com.stonewu.fusion.mapper.media.EpisodeAudioJobMapper;
import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.media.MediaJobSubmissionMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.entity.media.EpisodeAudioJobEntity;
import com.stonewu.fusion.entity.media.MediaJobEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.media.model.MediaException;
import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class EpisodeAudioReviewService {
    public enum Action {RECHECK, RECORD_RECEIPT, RETRY_INGEST, CONFIRM_FINISHED}

    public record Review(@NotNull @PositiveOrZero Long expectedRevision,
                         @NotBlank @Pattern(regexp = "[a-f0-9]{64}") String expectedFactsHash, @NotNull Action action,
                         @NotBlank @Size(max = 2000) String note, @Size(max = 16) String step,
                         @Size(max = 200) String providerTaskId, boolean confirmedEnded,
                         @NotNull @Valid GenerationClientScope scope) {
    }

    public record Step(String step, String providerTaskId, LocalDateTime submittedAt, boolean unresolved) {
    }

    public record View(String jobId, long revision, String status, boolean cancelRequested, boolean capacityHeld,
                       List<Step> steps, List<Action> actions, ObjectNode lastReview, String factsHash) {
    }

    private final EpisodeAudioJobMapper audioJobs;
    private final MediaJobMapper mediaJobs;
    private final MediaJobSubmissionMapper submissions;
    private final UserMapper users;
    private final ProjectMapper projects;
    private final MediaJobRepository jobs;
    private final EpisodeAudioJobs audio;
    private final CanvasAccessPolicy access;
    private final CanvasJson json;
    private final ApiConfigService configs;
    private final GenerationOutbox outbox;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View detail(String id, CanvasActor actor) {
        var job = require(id, actor, false);
        var pending = audio.unresolved(id);
        var steps = List.of("music", "sfx").stream().flatMap(step -> jobs.submission(id, step).stream()
                .map(value -> new Step(step, value.providerId(), value.createdAt(), pending.contains(step)))).toList();
        String review = audioJobs.selectById(id).getReviewJson();
        var publicReview = review == null ? null : json.read(review, ObjectNode.class);
        if (publicReview != null) {
            publicReview.remove("requestHash");
        }
        List<Action> actions = List.of();
        if (job.context().actor().equals(actor)) {
            try {
                access.requireProjectEditor(job.projectId(), actor);
                actions = actions(job);
            } catch (CanvasException forbidden) {
                if (forbidden.getCode() != 403) {
                    throw forbidden;
                }
            }
        }
        return new View(id, job.revision(), job.status(), job.cancelRequested(), audio.reserved(id), steps, actions,
                publicReview, factsHash(job));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public View review(String id, Review request, CanvasActor actor) {
        request.scope().require(actor);
        var job = require(id, actor, true);
        String hash = json.hash(request);
        String previous = audioJobs.lock(id).getReviewJson();
        if (previous != null && hash.equals(json.read(previous, ObjectNode.class).path("requestHash").asText())) {
            return detail(id, actor);
        }
        // 回执可由原调用迟到补记，并不推进任务版本；决定必须绑定操作者实际看到的证据。
        submissions.lockAttempts(id);
        if (job.revision() != request.expectedRevision() || !factsHash(job).equals(request.expectedFactsHash())
                || !actions(job).contains(request.action())) {
            throw new MediaException(409, "MEDIA_REVIEW_CONFLICT", "合成任务已变化，请刷新后再核实");
        }
        if (request.note() == null || request.note().isBlank()) {
            throw MediaException.invalid("请记录核实依据");
        }
        if (request.action() == Action.CONFIRM_FINISHED) {
            if (!request.confirmedEnded()) {
                throw MediaException.invalid("请明确确认所有待核实的远端请求均已结束");
            }
            audioJobs.update(Wrappers.<EpisodeAudioJobEntity>lambdaUpdate().eq(EpisodeAudioJobEntity::getJobId, id)
                    .set(EpisodeAudioJobEntity::getResolvedStepsJson, json.write(audio.unresolved(id)))
                    .set(EpisodeAudioJobEntity::getPermitHeld, false));
            mediaJobs.update(Wrappers.<MediaJobEntity>lambdaUpdate().eq(MediaJobEntity::getId, id)
                    .set(MediaJobEntity::getCancelRequested, true));
        } else {
            var settings = json.read(json.write(job.input()), ComposeSnapshot.class).bgm();
            var api = configs.getById(settings.apiConfigId());
            if (api == null || !Integer.valueOf(1).equals(api.getStatus()) || !json.hash(api)
                    .equals(settings.configHash())) {
                throw new MediaException(409, "MEDIA_AUDIO_CONFIG_CHANGED", "请恢复原音频服务配置后再继续查询或登记");
            }
            if (request.action() == Action.RECORD_RECEIPT) {
                if (request.step() == null || !List.of("music", "sfx")
                        .contains(request.step()) || request.providerTaskId() == null
                        || !request.providerTaskId().matches("[A-Za-z0-9_.:-]{1,200}")) {
                    throw MediaException.invalid("请选择原步骤并填写供应商任务标识，不能填写 URL");
                }
                var intent = jobs.submission(id, request.step())
                        .orElseThrow(() -> MediaException.invalid("此步骤尚未提交"));
                if (intent.providerId() != null) {
                    throw MediaException.invalid("此步骤已有不可变受理回执");
                }
                jobs.acknowledge(intent.attempt(), request.providerTaskId());
            }
        }
        var decision = json.object(request).put("requestHash", hash).put("actorUserId", actor.userId())
                .put("recordedAt", jobs.now().toString());
        audioJobs.update(Wrappers.<EpisodeAudioJobEntity>lambdaUpdate().eq(EpisodeAudioJobEntity::getJobId, id)
                .set(EpisodeAudioJobEntity::getReviewJson, json.write(decision))
                .set(EpisodeAudioJobEntity::getConsecutiveErrors, 0));
        mediaJobs.resumeReviewedAudio(id);
        var current = jobs.get(id);
        var event = decision.deepCopy().put("jobId", id).put("revision", current.revision());
        event.remove("requestHash");
        outbox.enqueue("episode-audio:" + id + ":" + current.revision(), "EPISODE_AUDIO_REVIEW", id, event);
        return detail(id, actor);
    }

    private String factsHash(MediaJob job) {
        return json.hash(Map.of("jobId", job.id(), "revision", job.revision(), "status", job.status(),
                "cancelRequested", job.cancelRequested(), "audio", audioJobs.selectMaps(
                        Wrappers.<EpisodeAudioJobEntity>lambdaQuery()
                                .select(EpisodeAudioJobEntity::getPermitHeld, EpisodeAudioJobEntity::getMusicResultHash,
                                        EpisodeAudioJobEntity::getSfxResultHash,
                                        EpisodeAudioJobEntity::getResolvedStepsJson)
                                .eq(EpisodeAudioJobEntity::getJobId, job.id())).getFirst(), "submissions",
                submissions.selectMaps(Wrappers.<MediaJobSubmissionEntity>lambdaQuery()
                        .select(MediaJobSubmissionEntity::getStep, MediaJobSubmissionEntity::getAttemptId,
                                MediaJobSubmissionEntity::getProviderTaskId)
                        .eq(MediaJobSubmissionEntity::getJobId, job.id())
                        .orderByAsc(MediaJobSubmissionEntity::getStep))));
    }

    private MediaJob require(String id, CanvasActor actor, boolean edit) {
        var job = jobs.get(id);
        if (!job.kind().equals("EPISODE_COMPOSE") || job.projectId() == null
                || !audioJobs.exists(
                Wrappers.<EpisodeAudioJobEntity>lambdaQuery().eq(EpisodeAudioJobEntity::getJobId, id))) {
            throw new MediaException(404, "EPISODE_AUDIO_NOT_FOUND", "此合成没有音频任务");
        }
        if (edit) {
            users.lockEnabled(actor.userId());
            projects.lock(job.projectId());
            access.requireProjectEditor(job.projectId(), actor);
            if (!job.context().actor().equals(actor)) {
                throw new MediaException(403, "MEDIA_REVIEW_DENIED", "只能核实原账号与团队发起的合成任务");
            }
            mediaJobs.lock(id);
        } else {
            access.requireProject(job.projectId(), actor);
        }
        return jobs.get(id);
    }

    private List<Action> actions(MediaJob job) {
        if (!job.status().equals("NEEDS_REVIEW")) {
            return List.of();
        }
        var result = new ArrayList<Action>();
        var pending = audio.unresolved(job.id());
        if (pending.stream().anyMatch(step -> jobs.submission(job.id(), step).orElseThrow().providerId() == null)) {
            result.add(Action.RECORD_RECEIPT);
        } else if (!pending.isEmpty()) {
            result.add(Action.RECHECK);
        }
        if (!job.cancelRequested() && pending.isEmpty() && audio.hasSuccessfulResult(job.id())) {
            result.add(Action.RETRY_INGEST);
        }
        result.add(Action.CONFIRM_FINISHED);
        return List.copyOf(result);
    }
}
