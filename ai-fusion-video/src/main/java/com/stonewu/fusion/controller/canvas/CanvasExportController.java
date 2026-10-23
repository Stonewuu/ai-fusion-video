package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportData.Options;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportData.Preview;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportData.Submit;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportData.Summary;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportService;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportTickets;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/canvases/{canvasId}/exports")
@RequiredArgsConstructor
public class CanvasExportController {
    private final CanvasActorResolver actors;
    private final CanvasExportService exports;
    private final CanvasExportTickets tickets;

    @PostMapping("/preview")
    public CommonResult<Preview> preview(@PathVariable String canvasId, @Valid @RequestBody Options options) {
        return CommonResult.success(exports.preview(canvasId, options, actors.current()));
    }

    @PostMapping
    public CommonResult<Summary> submit(@PathVariable String canvasId, @Valid @RequestBody Submit request) {
        return CommonResult.success(exports.submit(canvasId, request, actors.current()));
    }

    @GetMapping
    public CommonResult<List<Summary>> list(@PathVariable String canvasId, @RequestParam(defaultValue = "0") int page) {
        return CommonResult.success(exports.list(canvasId, page, actors.current()));
    }

    @GetMapping("/{id}")
    public CommonResult<Summary> get(@PathVariable String canvasId, @PathVariable String id) {
        return CommonResult.success(exports.get(canvasId, id, actors.current()));
    }

    @PostMapping("/{id}/cancel")
    public CommonResult<Boolean> cancel(@PathVariable String canvasId, @PathVariable String id) {
        exports.cancel(canvasId, id, actors.current());
        return CommonResult.success(true);
    }

    @PostMapping("/{id}/access")
    public CommonResult<CanvasExportTickets.Issued> download(@PathVariable String canvasId, @PathVariable String id) {
        return CommonResult.success(tickets.issue(canvasId, id, actors.current()));
    }
}
