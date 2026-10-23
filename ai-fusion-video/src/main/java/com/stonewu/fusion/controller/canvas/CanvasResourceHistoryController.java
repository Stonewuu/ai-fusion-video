package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceHistoryService;
import com.stonewu.fusion.service.resource.ResourceRestorePlanner;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.ResourceRestoreCommand;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/canvases/{canvasId}/resource-history")
@RequiredArgsConstructor
public class CanvasResourceHistoryController {
    private final CanvasActorResolver actors;
    private final ResourceHistoryService history;
    private final ResourceRestorePlanner planner;
    private final ResourceCommandService commands;

    @GetMapping
    public CommonResult<ResourceHistoryService.Page> list(@PathVariable String canvasId,
                                                          @RequestParam(required = false) String before,
                                                          @RequestParam(defaultValue = "40") int limit) {
        return CommonResult.success(history.list(canvasId, before, limit, actors.current()));
    }

    @GetMapping("/{commandId}")
    public CommonResult<ResourceHistoryService.Detail> detail(@PathVariable String canvasId,
                                                              @PathVariable String commandId) {
        return CommonResult.success(history.detail(canvasId, commandId, actors.current()));
    }

    @GetMapping("/{commandId}/restore-preview")
    public CommonResult<ResourceRestorePlanner.Plan> preview(@PathVariable String canvasId,
                                                             @PathVariable String commandId) {
        return CommonResult.success(planner.preview(canvasId, commandId, actors.current()));
    }

    @PostMapping("/restore")
    public CommonResult<ResourceCommandReceipt> restore(@PathVariable String canvasId,
                                                        @Valid @RequestBody ResourceRestoreCommand command) {
        return CommonResult.success(commands.execute(command, ResourceCommandContext.user(actors.current(), canvasId)));
    }
}
