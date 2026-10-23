package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.View;
import com.stonewu.fusion.service.canvas.run.CanvasRunService;
import com.stonewu.fusion.service.canvas.run.writeback.CanvasWritebackService;
import com.stonewu.fusion.service.canvas.run.writeback.CanvasWritebackService.Decision;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class CanvasWritebackController {
    private final CanvasActorResolver actors;
    private final CanvasWritebackService writebacks;
    private final CanvasRunService runs;

    @PostMapping("/api/canvas-runs/{runId}/nodes/{nodeId}/writeback")
    public CommonResult<View> decide(@PathVariable String runId, @PathVariable String nodeId,
                                     @Valid @RequestBody Decision request) {
        var actor = actors.current();
        writebacks.decide(runId, nodeId, request, actor);
        return CommonResult.success(runs.view(runId, actor));
    }
}
