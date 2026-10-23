package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.ai.vo.ToolConfirmationReqVO;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeQueries;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

@RestController
@RequiredArgsConstructor
public class CanvasAgentNodeController {
    private final CanvasAgentNodeQueries queries;
    private final CanvasActorResolver actors;

    @GetMapping("/api/canvas-runs/{runId}/nodes/{nodeId}/agent")
    public CommonResult<CanvasAgentNodeQueries.View> detail(@PathVariable String runId, @PathVariable String nodeId) {
        return CommonResult.success(queries.detail(runId, nodeId, actors.current()));
    }

    @PostMapping("/api/canvas-runs/{runId}/nodes/{nodeId}/agent/confirm")
    public Mono<CommonResult<Boolean>> confirm(@PathVariable String runId, @PathVariable String nodeId,
                                               @RequestBody ToolConfirmationReqVO request) {
        return queries.confirm(runId, nodeId, request, actors.current()).thenReturn(CommonResult.success(true));
    }
}
