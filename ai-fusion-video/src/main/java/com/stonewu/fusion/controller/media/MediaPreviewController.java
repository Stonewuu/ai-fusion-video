package com.stonewu.fusion.controller.media;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.media.preview.MediaPreviewData;
import com.stonewu.fusion.service.media.preview.MediaPreviewQueries;
import com.stonewu.fusion.service.media.preview.MediaPreviewService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/media-resources/{id}/preview")
@RequiredArgsConstructor
public class MediaPreviewController {
    private final CanvasActorResolver actors;
    private final MediaPreviewQueries queries;
    private final MediaPreviewService previews;

    @GetMapping
    public CommonResult<MediaPreviewData.View> get(@PathVariable String id) {
        return CommonResult.success(queries.get(id, actors.current()));
    }

    @PostMapping
    public CommonResult<MediaPreviewData.View> ensure(@PathVariable String id,
                                                      @Valid @RequestBody MediaPreviewData.Request request) {
        return CommonResult.success(previews.ensure(id, request, actors.current()));
    }
}
