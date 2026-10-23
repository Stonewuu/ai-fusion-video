package com.stonewu.fusion.service.canvas.run.mcp;

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
public class CanvasMcpExecutor implements Executor {
    private final CanvasMcpJobs tasks;
    private final CanvasMcpRepository repository;
    private final CanvasRunSources sources;

    @Override
    public Set<CanvasNodeKind> kinds() {
        return Set.of(CanvasNodeKind.MCP_TOOL);
    }

    @Override
    public Outcome step(Run run, NodeRun node, Claim claim) {
        if (node.externalId() == null) {
            return new Outcome(State.RUNNING, Ports.empty(), "MCP", tasks.submit(run, node), null);
        }
        var job = repository.get(node.externalId(), false);
        if (!job.nodeRunId().equals(node.id()) || !job.runId().equals(run.id())) {
            throw CanvasRunException.invalid("MCP 调用与节点关联不一致");
        }
        State state = switch (job.status()) {
            case "SUCCEEDED" -> State.SUCCEEDED;
            case "FAILED" -> State.FAILED;
            case "CANCELLED" -> State.CANCELLED;
            case "UNKNOWN_SUBMISSION" -> State.UNKNOWN_SUBMISSION;
            case "NEEDS_REVIEW" -> State.NEEDS_REVIEW;
            default -> job.cancelled() ? State.CANCEL_REQUESTED : State.RUNNING;
        };
        var output = Ports.empty();
        if (state == State.SUCCEEDED) {
            var value = job.result().path("output");
            output = new Ports(
                    Map.of("text", List.of(sources.text(value.isTextual() ? value.asText() : value.toString()))));
        }
        return new Outcome(state, output, "MCP", job.id(), job.error());
    }

    @Override
    public void cancel(Run run, NodeRun node) {
        if (node.externalId() != null) {
            tasks.cancel(node.externalId(), run.actor());
        }
    }
}
