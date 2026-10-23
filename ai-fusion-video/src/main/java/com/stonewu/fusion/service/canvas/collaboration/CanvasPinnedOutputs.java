package com.stonewu.fusion.service.canvas.collaboration;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasEdge;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.State;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Objects;
import java.util.function.UnaryOperator;

@Component
@RequiredArgsConstructor
public class CanvasPinnedOutputs {
    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasAccessPolicy access;
    private final CanvasRunRepository runs;
    private final CanvasRunSources sources;
    private final CanvasValueSnapshots snapshots;

    public Value edge(String canvasId, String edgeId, CanvasActor actor) {
        var document = documents.require(canvasId, false);
        access.requireView(document, actor);
        var edge = graphs.load(document).edges().stream()
                .filter(value -> value.id().equals(edgeId) && value.output() != null).findFirst()
                .orElseThrow(() -> new CanvasRunException(404, "FIXED_RESULT_NOT_FOUND", "此连线没有固定结果"));
        return resolve(canvasId, edge, actor);
    }

    public Value resolve(String canvasId, CanvasEdge edge, CanvasActor actor) {
        access.requireView(documents.require(canvasId, false), actor);
        var reference = edge.output();
        Value value;
        if (reference.snapshotId() != null) {
            var saved = snapshots.get(reference.snapshotId());
            if (!saved.canvasId().equals(canvasId) || !saved.nodeId().equals(edge.sourceNodeId()) || !saved.portId()
                    .equals(edge.sourcePort())) {
                throw CanvasRunException.invalid("固定快照不属于此画布节点和端口");
            }
            value = saved.value();
        } else {
            var original = runs.node(reference.nodeRunId(), false);
            if (!runs.run(original.runId(), false).canvasId().equals(canvasId) || !original.nodeId()
                    .equals(edge.sourceNodeId())
                    || original.status() != State.SUCCEEDED || original.parentNodeRunId() != null) {
                throw CanvasRunException.invalid("只能固定本画布对应节点的一次成功输出；批量子项请使用汇总结果");
            }
            value = original.outputs().at(edge.sourcePort()).stream()
                    .filter(item -> item.id().equals(reference.outputId())).findFirst()
                    .orElseThrow(() -> CanvasRunException.invalid("固定的上游输出不存在"));
        }
        if (!value.id().equals(reference.outputId()) || !Objects.equals(value.mediaId(), reference.resourceId())) {
            throw CanvasRunException.invalid("固定结果的内容或媒体标识不匹配");
        }
        sources.authorize(value, actor, true);
        return value;
    }

    public CanvasValueSnapshots.Snapshot read(String canvasId, String id, CanvasActor actor) {
        access.requireView(documents.require(canvasId, false), actor);
        var snapshot = snapshots.get(id);
        if (!snapshot.canvasId().equals(canvasId)) {
            throw new CanvasRunException(403, "FIXED_RESULT_SCOPE", "固定结果不属于此画布");
        }
        sources.authorize(snapshot.value(), actor, true);
        return snapshot;
    }

    public CanvasGraph copy(String sourceCanvas, CanvasGraph source, String targetCanvas, CanvasGraph copied,
                            CanvasActor actor, UnaryOperator<Value> remap) {
        if (source.edges().size() != copied.edges().size()) {
            throw new IllegalArgumentException("复制前后连线数量不一致");
        }
        var edges = new ArrayList<CanvasEdge>();
        for (int index = 0; index < source.edges().size(); index++) {
            var original = source.edges().get(index);
            var target = copied.edges().get(index);
            if (original.output() == null) {
                edges.add(target);
                continue;
            }
            var value = remap.apply(resolve(sourceCanvas, original, actor));
            var output = snapshots.copy(sourceCanvas, original.output(), targetCanvas, target.sourceNodeId(),
                    target.sourcePort(), value, actor);
            edges.add(new CanvasEdge(target.id(), target.kind(), target.sourceNodeId(), target.sourcePort(),
                    target.targetNodeId(),
                    target.targetPort(), target.inputIndex(), output, target.version()));
        }
        return new CanvasGraph(copied.schemaVersion(), copied.projectId(), copied.nodes(), edges);
    }
}
