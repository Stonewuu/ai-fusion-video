package com.stonewu.fusion.service.canvas.delivery;

import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.CanvasRunExecutors;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Set;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

@Component
@RequiredArgsConstructor
public class CanvasExportExecutor implements CanvasRunExecutors.Executor {
    private final CanvasExportService exports;
    private final CanvasExportRepository jobs;

    @Override
    public Set<CanvasNodeKind> kinds() {
        return Set.of(CanvasNodeKind.EXPORT);
    }

    @Override
    public CanvasRunExecutors.Outcome step(Run run, NodeRun node, Claim claim) {
        if (node.externalId() == null) {
            return new CanvasRunExecutors.Outcome(State.RUNNING, Ports.empty(), "EXPORT", exports.submitNode(run, node),
                    null);
        }
        var job = jobs.get(node.externalId(), false);
        if (!node.id().equals(job.nodeRunId())) {
            throw new IllegalStateException("交付任务不属于此运行节点");
        }
        State state = switch (job.status()) {
            case "SUCCEEDED" -> State.SUCCEEDED;
            case "FAILED" -> State.FAILED;
            case "CANCELLED" -> State.CANCELLED;
            case "QUEUED", "RUNNING" -> State.RUNNING;
            default -> throw new IllegalStateException("未知交付状态");
        };
        return new CanvasRunExecutors.Outcome(state, Ports.empty(), "EXPORT", job.id(), job.error());
    }

    @Override
    public void cancel(Run run, NodeRun node) {
        if (node.externalId() != null) {
            exports.cancel(run.canvasId(), node.externalId(), run.actor());
        }
    }
}
