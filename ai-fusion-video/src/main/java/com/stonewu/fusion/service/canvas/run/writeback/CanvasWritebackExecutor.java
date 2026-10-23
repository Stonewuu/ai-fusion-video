package com.stonewu.fusion.service.canvas.run.writeback;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Set;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;
import static com.stonewu.fusion.service.canvas.run.CanvasRunExecutors.Executor;
import static com.stonewu.fusion.service.canvas.run.CanvasRunExecutors.Outcome;

@Component
@RequiredArgsConstructor
public class CanvasWritebackExecutor implements Executor {
    private final CanvasWritebackPreview preview;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;

    @Override
    public Set<CanvasNodeKind> kinds() {
        return Set.of(CanvasNodeKind.WRITEBACK);
    }

    @Override
    public Outcome step(Run run, NodeRun node, Claim claim) {
        access.requireEdit(documents.require(run.canvasId(), false), run.actor());
        node.checkpoint().set("writeback", preview.prepare(run, node));
        return new Outcome(State.WAITING_CONFIRMATION, Ports.empty(), "WRITEBACK", null, null);
    }
}
