package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.GenerationKind;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;

@Component
@RequiredArgsConstructor
public class GenerationJobQueries {
    private final GenerationJobRepository jobs;
    private final GenerationExecutionContexts access;
    private final GenerationAttemptRepository attempts;
    private final GenerationReviewPolicy reviewPolicy;

    public Summary summary(String id, CanvasActor actor) {
        var value = jobs.submission(id);
        access.requireRead(value.kind(), value.taskId(), actor);
        return new Summary(value.id(), value.kind(), value.state(), value.model().getName());
    }

    public View get(String id, CanvasActor actor) {
        var value = jobs.submission(id);
        access.requireRead(value.kind(), value.taskId(), actor);
        return view(value, actor);
    }

    public View task(GenerationKind kind, long taskId, CanvasActor actor) {
        access.requireRead(kind, taskId, actor);
        return view(jobs.byTask(kind, taskId).orElseThrow(() ->
                new GenerationJobException(409, "GENERATION_LEGACY_RECORD", "此历史任务没有可恢复执行记录")), actor);
    }

    private View view(GenerationSubmission submission, CanvasActor actor) {
        boolean canControl = submission.userId() == actor.userId();
        var children = jobs.jobs(submission.id()).stream().map(job -> {
            var history = attempts.forJob(job.id());
            return new Item(job.id(), job.itemIds(), job.itemIndex(),
                    job.state(), job.phase(), job.providerTaskId(), job.cancelRequested(), job.cancelSent(),
                    job.permitHeld(),
                    job.errorCode(), job.errorMessage(), job.revision(), job.deadline().toInstant(ZoneOffset.UTC),
                    job.result() != null
                            && job.result().state() == RemoteGenerationResult.State.SUCCEEDED,
                    canControl ? reviewPolicy.allowed(job, history) : List.of(),
                    history.stream()
                            .map(attempt -> new AttemptView(attempt.id(), attempt.number(), attempt.providerTaskId(),
                                    attempt.submitted(), attempt.acknowledged(), attempt.resolution())).toList());
        }).toList();
        return new View(submission.id(), submission.kind(), submission.taskId(), submission.publicTaskId(),
                submission.state(), submission.cancelRequested(), submission.revision(), submission.model().getId(),
                submission.model().getName(), submission.createdAt().toInstant(ZoneOffset.UTC), canControl, children);
    }

    public record Summary(String id, GenerationKind kind, String state, String modelName) {
    }

    public record View(String id, GenerationKind kind, long recordId, String taskId, String state,
                       boolean cancelRequested, long revision, long modelId, String modelName,
                       Instant createdAt, boolean canControl, List<Item> items) {
    }

    public record Item(String id, List<Long> itemIds, int index, String state, String phase, String providerTaskId,
                       boolean cancelRequested, boolean cancelSent, boolean remoteCapacityHeld,
                       String errorCode, String errorMessage, long revision, Instant deadline, boolean resultReceived,
                       List<GenerationJobCommands.Action> availableActions, List<AttemptView> attempts) {
    }

    public record AttemptView(String id, int number, String providerTaskId, boolean submitted, boolean acknowledged,
                              String resolution) {
    }
}
