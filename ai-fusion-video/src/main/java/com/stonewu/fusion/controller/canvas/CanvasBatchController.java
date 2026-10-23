package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.View;
import com.stonewu.fusion.service.canvas.run.CanvasRunService;
import com.stonewu.fusion.service.canvas.run.batch.CanvasBatchReviewService;
import com.stonewu.fusion.service.canvas.run.batch.CanvasBatchReviewService.Review;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class CanvasBatchController {
    private final CanvasBatchReviewService batches;
    private final CanvasRunService runs;
    private final CanvasActorResolver actors;

    @PostMapping("/api/canvas-runs/{runId}/nodes/{nodeId}/batch/review")
    public CommonResult<View> review(@PathVariable String runId, @PathVariable String nodeId,
                                     @Valid @RequestBody Review request) {
        var actor = actors.current();
        batches.review(runId, nodeId, request, actor);
        return CommonResult.success(runs.view(runId, actor));
    }
}
