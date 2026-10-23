package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.collaboration.CanvasPinnedOutputs;
import com.stonewu.fusion.service.canvas.collaboration.CanvasValueSnapshots.Snapshot;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class CanvasValueSnapshotController {
    private final CanvasPinnedOutputs outputs;
    private final CanvasActorResolver actors;

    @GetMapping("/api/canvases/{canvasId}/value-snapshots/{id}")
    public CommonResult<Snapshot> read(@PathVariable String canvasId, @PathVariable String id) {
        return CommonResult.success(outputs.read(canvasId, id, actors.current()));
    }

    @GetMapping("/api/canvases/{canvasId}/fixed-outputs/{edgeId}")
    public CommonResult<Value> edge(@PathVariable String canvasId, @PathVariable String edgeId) {
        return CommonResult.success(outputs.edge(canvasId, edgeId, actors.current()));
    }
}
