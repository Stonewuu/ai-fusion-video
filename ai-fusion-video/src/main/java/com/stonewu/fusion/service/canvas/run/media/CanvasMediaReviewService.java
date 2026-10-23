package com.stonewu.fusion.service.canvas.run.media;

import com.stonewu.fusion.mapper.canvas.CanvasMediaJobMapper;
import com.stonewu.fusion.entity.canvas.CanvasMediaJobEntity;
import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@Service
@RequiredArgsConstructor
public class CanvasMediaReviewService {
    public enum Action {RECHECK, RETRY_INGEST, RECORD_RECEIPT, CONFIRM_FINISHED}

    public record Review(@NotNull Long expectedRevision, @NotNull Action action, @Size(max = 2000) String note,
                         @Size(max = 200) String providerTaskId, @NotNull @Valid GenerationClientScope scope) {
    }

    public record View(String id, String status, String phase, String error, long revision,
                       String providerTaskId, boolean remoteCapacityHeld, boolean cancelRequested,
                       List<Action> actions) {
    }

    private final CanvasRunRepository runs;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final MediaJobRepository jobs;
    private final CanvasMediaJobs tasks;
    private final GenerationOutbox outbox;
    private final CanvasMediaJobMapper canvasJobs;
    private final MediaJobMapper mediaJobs;
    private final CanvasJson json;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View detail(String runId, String nodeId, CanvasActor actor) {
        var run = runs.run(runId, false);
        var permissions = access.requireView(documents.require(run.canvasId(), false), actor);
        var job = task(runId, nodeId);
        var submission = jobs.submission(job.id(), "audio");
        boolean held = Boolean.TRUE.equals(
                canvasJobs.selectById(job.id()).getPermitHeld());
        return new View(job.id(), job.status(), job.phase(), job.error(), job.revision(),
                submission.map(MediaJobRepository.Submission::providerId).orElse(null), held, job.cancelRequested(),
                run.actor().equals(actor) && permissions.canExecute() ? actions(job) : List.of());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = {"canvasRun", "canvasMediaJob"}, allEntries = true)
    public View review(String runId, String nodeId, Review request, CanvasActor actor) {
        request.scope().require(actor);
        runs.lockActor(actor);
        var observed = runs.run(runId, false);
        if (!access.requireView(documents.require(observed.canvasId(), true), actor).canExecute() || !observed.actor()
                .equals(actor)) {
            throw new CanvasRunException(403, "MEDIA_REVIEW_DENIED", "只能核实原账号与团队发起的媒体任务");
        }
        runs.run(runId, true);
        var job = task(runId, nodeId);
        mediaJobs.lock(job.id());
        job = jobs.get(job.id());
        if (job.revision() != request.expectedRevision() || !actions(job).contains(request.action())) {
            throw CanvasRunException.conflict("媒体任务状态已变化，请刷新后再处理");
        }
        if (request.note() == null || request.note().isBlank()) {
            throw CanvasRunException.invalid("请记录此次核实依据");
        }
        if (request.action() == Action.CONFIRM_FINISHED) {
            mediaJobs.reviewCanvasMedia(job.cancelRequested() ? "CANCELLED" : "FAILED", request.note(), job.id());
            canvasJobs.update(Wrappers.<CanvasMediaJobEntity>lambdaUpdate().eq(CanvasMediaJobEntity::getJobId, job.id())
                    .set(CanvasMediaJobEntity::getPermitHeld, false));
        } else {
            if (request.action() == Action.RECORD_RECEIPT) {
                if (request.providerTaskId() == null || !request.providerTaskId().matches("[A-Za-z0-9_.:-]{1,200}")) {
                    throw CanvasRunException.invalid("请输入供应商原任务标识，不能填写 URL");
                }
                var intent = jobs.submission(job.id(), "audio").orElseThrow();
                jobs.acknowledge(intent.attempt(), request.providerTaskId());
            }
            tasks.api(tasks.input(job));
            mediaJobs.resumeCanvasMedia(job.id());
            canvasJobs.update(Wrappers.<CanvasMediaJobEntity>lambdaUpdate().eq(CanvasMediaJobEntity::getJobId, job.id())
                    .set(CanvasMediaJobEntity::getConsecutiveErrors, 0));
        }
        var current = jobs.get(job.id());
        outbox.enqueue("canvas-media:" + job.id() + ":" + current.revision(), "CANVAS_MEDIA_REVIEW", job.id(),
                json.object(
                        Map.of("jobId", job.id(), "revision", current.revision(), "action", request.action(), "note",
                                request.note(),
                                "actorUserId", actor.userId(), "providerTaskId",
                                Objects.toString(request.providerTaskId(), ""))));
        return detail(runId, nodeId, actor);
    }

    private MediaJob task(String runId, String nodeId) {
        var node = runs.node(nodeId, false);
        if (!node.runId().equals(runId) || !"MEDIA".equals(node.externalKind()) || node.externalId() == null) {
            throw CanvasRunException.invalid("此节点没有对应媒体任务");
        }
        return jobs.get(node.externalId());
    }

    private List<Action> actions(MediaJob job) {
        if (!List.of("NEEDS_REVIEW", "UNKNOWN_SUBMISSION").contains(job.status())) {
            return List.of();
        }
        var result = new ArrayList<Action>();
        var intent = jobs.submission(job.id(), "audio");
        if (intent.isPresent()) {
            if (intent.get().providerId() == null) {
                result.add(Action.RECORD_RECEIPT);
            } else if (tasks.received(job.id()) == null) {
                result.add(Action.RECHECK);
            }
        }
        if (!job.cancelRequested() && tasks.received(job.id()) != null) {
            result.add(Action.RETRY_INGEST);
        }
        result.add(Action.CONFIRM_FINISHED);
        return List.copyOf(result);
    }
}
