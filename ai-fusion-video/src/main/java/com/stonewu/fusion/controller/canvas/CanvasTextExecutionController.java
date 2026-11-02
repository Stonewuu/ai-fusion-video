package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.run.text.CanvasTextReviewService;
import com.stonewu.fusion.service.canvas.run.text.CanvasTextReviewService.View;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
public class CanvasTextExecutionController {
    private final CanvasTextReviewService service;
    private final CanvasActorResolver actors;

    @GetMapping("/api/canvas-runs/{runId}/nodes/{nodeId}/text")
    public CommonResult<View> detail(@PathVariable String runId, @PathVariable String nodeId) {
        return CommonResult.success(service.detail(runId, nodeId, actors.current()));
    }

}
