package com.stonewu.fusion.controller.media;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.media.importing.MediaImportData;
import com.stonewu.fusion.service.media.importing.MediaImportPlacementService;
import com.stonewu.fusion.service.media.importing.MediaImportQueries;
import com.stonewu.fusion.service.media.importing.MediaImportService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class MediaImportController {
    private final CanvasActorResolver actors;
    private final MediaImportQueries queries;
    private final MediaImportService imports;
    private final MediaImportPlacementService placement;

    @PostMapping("/canvases/{canvasId}/media-imports")
    public CommonResult<MediaImportData.View> submit(@PathVariable String canvasId,
                                                     @Valid @RequestBody MediaImportData.Submit request) {
        return CommonResult.success(
                imports.submit(canvasId, request, ResourceCommandContext.user(actors.current(), canvasId)));
    }

    @GetMapping("/canvases/{canvasId}/media-imports")
    public CommonResult<MediaImportData.Page> list(@PathVariable String canvasId,
                                                   @RequestParam(defaultValue = "0") int page) {
        return CommonResult.success(queries.list(canvasId, page, actors.current()));
    }

    @GetMapping("/canvases/{canvasId}/media-imports/requests/{requestId}")
    public CommonResult<MediaImportData.View> request(@PathVariable String canvasId, @PathVariable String requestId) {
        return CommonResult.success(queries.request(canvasId, requestId, actors.current()));
    }

    @GetMapping("/canvases/{canvasId}/media-imports/active")
    public CommonResult<List<MediaImportData.Summary>> active(@PathVariable String canvasId) {
        return CommonResult.success(queries.active(canvasId, actors.current()));
    }

    @PostMapping("/canvases/{canvasId}/media-imports/requests/{requestId}/withdraw")
    public CommonResult<MediaImportData.Withdrawal> withdraw(@PathVariable String canvasId,
                                                             @PathVariable String requestId,
                                                             @Valid @RequestBody MediaImportData.Control request) {
        return CommonResult.success(imports.withdraw(canvasId, requestId, request.scope(), actors.current()));
    }

    @GetMapping("/media-imports/{id}")
    public CommonResult<MediaImportData.View> get(@PathVariable String id) {
        return CommonResult.success(queries.get(id, actors.current()));
    }

    @PostMapping("/media-imports/{id}/cancel")
    public CommonResult<MediaImportData.View> cancel(@PathVariable String id,
                                                     @Valid @RequestBody MediaImportData.Control request) {
        return CommonResult.success(imports.cancel(id, request.scope(), actors.current()));
    }

    @PostMapping("/media-imports/{id}/retry")
    public CommonResult<MediaImportData.View> retry(@PathVariable String id,
                                                    @Valid @RequestBody MediaImportData.Retry request) {
        return CommonResult.success(imports.retry(id, request, actors.current()));
    }

    @PostMapping("/media-imports/{id}/place")
    public CommonResult<MediaImportData.Placed> place(@PathVariable String id,
                                                      @Valid @RequestBody MediaImportData.Place request) {
        return CommonResult.success(placement.place(id, request, actors.current()));
    }
}
