package com.stonewu.fusion.service.canvas.run.media;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.CanvasRunData;
import com.stonewu.fusion.service.canvas.run.CanvasRunExecutors;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Set;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;
import static com.stonewu.fusion.service.canvas.run.CanvasRunExecutors.Outcome;

@Component
@RequiredArgsConstructor
public class CanvasMediaExecutor implements CanvasRunExecutors.Executor {
    public static final Set<CanvasNodeKind> LOCAL = Set.of(CanvasNodeKind.FRAME_EXTRACT, CanvasNodeKind.IMAGE_TRANSFORM,
            CanvasNodeKind.VIDEO_COMPOSE, CanvasNodeKind.AUDIO_MIX);
    public static final Set<CanvasNodeKind> AUDIO = Set.of(CanvasNodeKind.MUSIC_GENERATION,
            CanvasNodeKind.SFX_GENERATION);
    private final CanvasMediaJobs tasks;
    private final MediaJobRepository jobs;
    private final CanvasJson json;

    @Override
    public Set<CanvasNodeKind> kinds() {
        return Set.of(CanvasNodeKind.FRAME_EXTRACT, CanvasNodeKind.IMAGE_TRANSFORM, CanvasNodeKind.VIDEO_COMPOSE,
                CanvasNodeKind.AUDIO_MIX, CanvasNodeKind.MUSIC_GENERATION, CanvasNodeKind.SFX_GENERATION);
    }

    @Override
    public Outcome step(Run run, NodeRun node, CanvasRunData.Claim claim) {
        if (node.externalId() == null) {
            return new Outcome(State.RUNNING, Ports.empty(), "MEDIA", tasks.submit(run, node), null);
        }
        var task = jobs.get(node.externalId());
        State state = switch (task.status()) {
            case "SUCCEEDED" -> State.SUCCEEDED;
            case "FAILED" -> State.FAILED;
            case "CANCELLED" -> State.CANCELLED;
            case "NEEDS_REVIEW" -> State.NEEDS_REVIEW;
            case "UNKNOWN_SUBMISSION" -> State.UNKNOWN_SUBMISSION;
            case "QUEUED", "WAITING", "RUNNING" -> task.cancelRequested() ? State.CANCEL_REQUESTED : State.RUNNING;
            default -> throw new IllegalStateException("画布媒体任务状态未登记：" + task.status());
        };
        Ports outputs = task.checkpoint().has("outputs") ? json.read(task.checkpoint().path("outputs").toString(),
                Ports.class) : Ports.empty();
        return new Outcome(state, outputs, "MEDIA", task.id(), task.error());
    }

    @Override
    public void cancel(Run run, NodeRun node) {
        if (node.externalId() != null) {
            tasks.cancel(node.externalId(), run.actor());
        }
    }
}
