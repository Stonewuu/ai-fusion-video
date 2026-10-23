package com.stonewu.fusion.service.canvas.run.text;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.run.text.CanvasTextData.Job;

@Service
@RequiredArgsConstructor
public class CanvasTextReviewService {
    public enum Action {USE_RECEIVED, CONFIRM_FINISHED}

    public record Review(@NotNull Long expectedRevision, @NotNull Action action,
                         @NotBlank @Size(max = 2000) String note,
                         @NotNull @Valid GenerationClientScope scope) {
    }

    public record View(String id, String status, String phase, String error, long revision, String receivedText,
                       boolean completeResponse, boolean remoteCapacityHeld, boolean cancelRequested,
                       List<Action> actions) {
    }

    private final CanvasRunRepository runs;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final CanvasTextRepository tasks;
    private final CanvasRunSources sources;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View detail(String runId, String nodeId, CanvasActor actor) {
        var run = runs.run(runId, false);
        var permissions = access.requireView(documents.require(run.canvasId(), false), actor);
        var job = task(runId, nodeId, false);
        job.input().inputs().all().forEach(value -> sources.authorize(value, actor, false));
        return new View(job.id(), job.status(), job.phase(), job.error(), job.revision(),
                job.result() == null ? null : job.result().path("text").asText(),
                job.result() != null && job.result().path("complete").asBoolean(), job.permitHeld(),
                job.cancelRequested(),
                run.actor().equals(actor) && permissions.canExecute() ? actions(job, run) : List.of());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = {"canvasRun", "canvasTextJob"}, allEntries = true)
    public View review(String runId, String nodeId, Review request, CanvasActor actor) {
        request.scope().require(actor);
        runs.lockActor(actor);
        var observed = runs.run(runId, false);
        if (!access.requireView(documents.require(observed.canvasId(), true), actor).canExecute() || !observed.actor()
                .equals(actor)) {
            throw new CanvasRunException(403, "TEXT_REVIEW_DENIED", "只能核实原账号和团队发起的文本任务");
        }
        var run = runs.run(runId, true);
        var job = task(runId, nodeId, true);
        if (job.revision() != request.expectedRevision() || !actions(job, run).contains(request.action())) {
            throw CanvasRunException.conflict("文本任务状态已变化，请刷新后再处理");
        }
        if (request.note() == null || request.note().isBlank()) {
            throw CanvasRunException.invalid("请记录此次核实依据");
        }
        var audit = Map.<String, Object>of("action", request.action(), "note", request.note(), "actorUserId",
                actor.userId());
        if (request.action() == Action.USE_RECEIVED) {
            job.input().inputs().all().forEach(value -> sources.authorize(value, actor, true));
            tasks.resume(job, audit);
        } else {
            tasks.finish(job, job.cancelRequested() || run.cancelRequested() ? "CANCELLED" : "FAILED",
                    "已人工核实原文本请求结束", false);
            tasks.event(tasks.get(job.id(), false), "CANVAS_TEXT_REVIEW", audit);
        }
        return detail(runId, nodeId, actor);
    }

    private List<Action> actions(Job job, Run run) {
        if (!List.of("NEEDS_REVIEW", "UNKNOWN_SUBMISSION").contains(job.status()) || run.status().terminal()) {
            return List.of();
        }
        var result = new ArrayList<Action>();
        if (job.result() != null && !job.result().path("text").asText().isBlank() && !job.cancelRequested()
                && !run.cancelRequested() && job.deadlineAt().isAfter(Instant.now()) && run.deadlineAt()
                .isAfter(Instant.now())) {
            result.add(Action.USE_RECEIVED);
        }
        result.add(Action.CONFIRM_FINISHED);
        return List.copyOf(result);
    }

    private Job task(String runId, String nodeId, boolean lock) {
        var node = runs.node(nodeId, false);
        if (!node.runId().equals(runId) || !"TEXT".equals(node.externalKind()) || node.externalId() == null) {
            throw CanvasRunException.invalid("此节点没有对应的文本任务");
        }
        return tasks.get(node.externalId(), lock);
    }
}
