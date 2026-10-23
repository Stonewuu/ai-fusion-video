package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.run.media.CanvasMediaReviewService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/canvas-runs/{runId}/nodes/{nodeId}/media")
@RequiredArgsConstructor
public class CanvasMediaRunController {
    private final CanvasMediaReviewService reviews;
    private final CanvasActorResolver actors;

    @GetMapping
    public CommonResult<CanvasMediaReviewService.View> detail(@PathVariable String runId, @PathVariable String nodeId) {
        return CommonResult.success(reviews.detail(runId, nodeId, actors.current()));
    }

    @PostMapping("/review")
    public CommonResult<CanvasMediaReviewService.View> review(@PathVariable String runId, @PathVariable String nodeId,
                                                              @Valid @RequestBody CanvasMediaReviewService.Review request) {
        return CommonResult.success(reviews.review(runId, nodeId, request, actors.current()));
    }
}
