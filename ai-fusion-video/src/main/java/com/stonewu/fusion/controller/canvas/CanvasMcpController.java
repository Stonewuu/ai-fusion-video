package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.security.SecurityUtils;
import com.stonewu.fusion.service.ai.agentscope.mcp.McpWorkflowTool;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpContracts;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.Contract;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpReviewService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
public class CanvasMcpController {
    private final CanvasMcpContracts contracts;
    private final CanvasMcpReviewService reviews;
    private final CanvasActorResolver actors;

    @GetMapping("/api/canvas-mcp/tools")
    public CommonResult<List<McpWorkflowTool>> tools() {
        return CommonResult.success(contracts.tools(SecurityUtils.requireCurrentUserId()));
    }

    @GetMapping("/api/canvas-mcp/contracts")
    public CommonResult<List<Contract>> list() {
        return CommonResult.success(contracts.list(SecurityUtils.requireCurrentUserId()));
    }

    @PostMapping("/api/canvas-mcp/contracts")
    public CommonResult<Contract> save(@Valid @RequestBody CanvasMcpContracts.Save request) {
        return CommonResult.success(contracts.save(SecurityUtils.requireCurrentUserId(), request));
    }

    @DeleteMapping("/api/canvas-mcp/contracts/{id}")
    public CommonResult<Boolean> remove(@PathVariable String id, @RequestParam long expectedVersion) {
        contracts.remove(id, expectedVersion, SecurityUtils.requireCurrentUserId());
        return CommonResult.success(true);
    }

    @GetMapping("/api/canvas-runs/{runId}/nodes/{nodeId}/mcp")
    public CommonResult<CanvasMcpReviewService.View> detail(@PathVariable String runId, @PathVariable String nodeId) {
        return CommonResult.success(reviews.detail(runId, nodeId, actors.current()));
    }

    @PostMapping("/api/canvas-runs/{runId}/nodes/{nodeId}/mcp/review")
    public CommonResult<CanvasMcpReviewService.View> review(@PathVariable String runId, @PathVariable String nodeId,
                                                            @Valid @RequestBody CanvasMcpReviewService.Review request) {
        return CommonResult.success(reviews.review(runId, nodeId, request, actors.current()));
    }
}
