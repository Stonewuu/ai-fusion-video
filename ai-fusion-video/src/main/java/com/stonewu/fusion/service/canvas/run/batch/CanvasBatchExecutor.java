package com.stonewu.fusion.service.canvas.run.batch;

import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.mapper.canvas.CanvasNodeRunMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Set;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;
import static com.stonewu.fusion.service.canvas.run.CanvasRunExecutors.Executor;
import static com.stonewu.fusion.service.canvas.run.CanvasRunExecutors.Outcome;

@Component
@RequiredArgsConstructor
public class CanvasBatchExecutor implements Executor {
    private final CanvasBatchItems batches;
    private final CanvasNodeRunMapper nodes;

    @Override
    public Set<CanvasNodeKind> kinds() {
        return Set.of(CanvasNodeKind.FOREACH);
    }

    @Override
    public Outcome step(Run run, NodeRun node, Claim claim) {
        if (node.externalId() == null) {
            batches.expand(run, node);
            return new Outcome(State.RUNNING, Ports.empty(), "BATCH", node.id(), null);
        }
        var items = batches.latest(node);
        if (items.stream().anyMatch(item -> !item.status().terminal())) {
            return new Outcome(run.cancelRequested() ? State.CANCEL_REQUESTED : State.RUNNING, Ports.empty(), "BATCH",
                    node.id(), null);
        }
        if (run.cancelRequested()) {
            return new Outcome(State.CANCELLED, Ports.empty(), "BATCH", node.id(), null);
        }
        long failures = items.stream().filter(item -> item.status() != State.SUCCEEDED).count();
        if (failures > 0) {
            if (nodes.hasUnknownRemote(items.stream().map(NodeRun::id).toList())) {
                return new Outcome(State.FAILED, Ports.empty(), "BATCH", node.id(),
                        "子项远端结果未知，已结束本批次，已有结果保留在子项详情中");
            }
            return new Outcome(State.WAITING_CONFIRMATION, Ports.empty(), "BATCH", node.id(),
                    failures + " 项未完整成功，请查看子项后选择下一步");
        }
        return new Outcome(State.SUCCEEDED, batches.results(node, items), "BATCH", node.id(), null);
    }
}
