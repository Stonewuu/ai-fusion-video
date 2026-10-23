package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.collaboration.CanvasPresence;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/canvases/{canvasId}/presence")
@RequiredArgsConstructor
public class CanvasPresenceController {
    private final CanvasActorResolver actors;
    private final CanvasPresence presence;

    @PostMapping
    public CommonResult<CanvasPresence.Snapshot> heartbeat(@PathVariable String canvasId,
                                                           @Valid @RequestBody CanvasPresence.Heartbeat request) {
        return CommonResult.success(presence.heartbeat(canvasId, request, actors.current()));
    }

    @DeleteMapping("/{sessionId}")
    public CommonResult<Boolean> leave(@PathVariable String canvasId, @PathVariable String sessionId) {
        presence.leave(canvasId, sessionId, actors.current());
        return CommonResult.success(true);
    }
}
