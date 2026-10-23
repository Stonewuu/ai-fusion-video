package com.stonewu.fusion.service.canvas.run.text;

import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.*;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunExecutors.Executor;
import com.stonewu.fusion.service.canvas.run.CanvasRunExecutors.Outcome;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.Set;

@Component
@RequiredArgsConstructor
public class CanvasTextExecutor implements Executor {
    private final CanvasTextJobs tasks;
    private final CanvasTextRepository repository;
    private final CanvasRunSources sources;

    @Override
    public Set<CanvasNodeKind> kinds() {
        return Set.of(CanvasNodeKind.TEXT_GENERATION);
    }

    @Override
    public Outcome step(Run run, NodeRun node, Claim claim) {
        if (node.externalId() == null) {
            return new Outcome(State.RUNNING, Ports.empty(), "TEXT", tasks.submit(run, node), null);
        }
        var job = repository.get(node.externalId(), false);
        if (!job.nodeRunId().equals(node.id())) {
            throw CanvasRunException.invalid("文本任务与节点运行关联不一致");
        }
        State state = switch (job.status()) {
            case "SUCCEEDED" -> State.SUCCEEDED;
            case "FAILED" -> State.FAILED;
            case "CANCELLED" -> State.CANCELLED;
            case "UNKNOWN_SUBMISSION" -> State.UNKNOWN_SUBMISSION;
            case "NEEDS_REVIEW" -> State.NEEDS_REVIEW;
            default -> job.cancelRequested() ? State.CANCEL_REQUESTED : State.RUNNING;
        };
        var output = state == State.SUCCEEDED
                ? new Ports(Map.of("text", List.of(sources.text(job.result().path("text").asText())))) : Ports.empty();
        return new Outcome(state, output, "TEXT", job.id(), job.error());
    }

    @Override
    public void cancel(Run run, NodeRun node) {
        if (node.externalId() != null) {
            tasks.cancel(node.externalId(), run.actor());
        }
    }
}
