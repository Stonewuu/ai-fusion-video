package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.run.CanvasActivityStatistics;
import com.stonewu.fusion.service.canvas.run.CanvasActivityStatistics.Activity;
import lombok.RequiredArgsConstructor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.time.Instant;

@RestController
@RequiredArgsConstructor
public class CanvasActivityController {
    private final CanvasActivityStatistics statistics;
    private final CanvasActorResolver actors;

    @GetMapping("/api/canvases/{canvasId}/activity-statistics")
    public CommonResult<Activity> read(@PathVariable String canvasId,
                                       @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) Instant from,
                                       @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) Instant until) {
        return CommonResult.success(statistics.read(canvasId, from, until, actors.current()));
    }
}
