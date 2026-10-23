package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.generation.vo.GenerationScopeReqVO;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.run.CanvasRunService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

@RestController
@RequiredArgsConstructor
public class CanvasRunController {
    private final CanvasRunService runs;
    private final CanvasActorResolver actors;

    @PostMapping("/api/canvases/{canvasId}/run-plans")
    public CommonResult<PlanView> plan(@PathVariable String canvasId, @Valid @RequestBody PlanRequest request) {
        return CommonResult.success(runs.plan(canvasId, request, actors.current()));
    }

    @PostMapping("/api/canvases/{canvasId}/runs")
    public CommonResult<View> start(@PathVariable String canvasId, @Valid @RequestBody Start request) {
        return CommonResult.success(
                runs.start(canvasId, request, ResourceCommandContext.user(actors.current(), canvasId)));
    }

    @GetMapping("/api/canvases/{canvasId}/runs")
    public CommonResult<List<Summary>> list(@PathVariable String canvasId, @RequestParam(defaultValue = "0") int page) {
        return CommonResult.success(runs.list(canvasId, page, actors.current()));
    }

    @GetMapping("/api/canvas-runs/{id}")
    public CommonResult<View> get(@PathVariable String id) {
        return CommonResult.success(runs.view(id, actors.current()));
    }

    @GetMapping("/api/canvases/{canvasId}/node-activities")
    public CommonResult<List<NodeActivity>> nodeActivities(@PathVariable String canvasId) {
        return CommonResult.success(runs.nodeActivities(canvasId, actors.current()));
    }

    @PostMapping("/api/canvas-runs/{id}/cancel")
    public CommonResult<View> cancel(@PathVariable String id, @Valid @RequestBody GenerationScopeReqVO request) {
        return CommonResult.success(runs.cancel(id, request.getScope(), actors.current()));
    }

    @PostMapping("/api/canvas-runs/{id}/nodes/{nodeId}/select")
    public CommonResult<View> select(@PathVariable String id, @PathVariable String nodeId,
                                     @Valid @RequestBody Selection request) {
        return CommonResult.success(runs.select(id, nodeId, request, actors.current()));
    }

    @PostMapping("/api/canvases/{canvasId}/run-requests/{requestId}/withdraw")
    public CommonResult<View> withdraw(@PathVariable String canvasId, @PathVariable String requestId,
                                       @Valid @RequestBody GenerationScopeReqVO request) {
        return CommonResult.success(runs.withdraw(canvasId, requestId, request.getScope(), actors.current()));
    }
}
