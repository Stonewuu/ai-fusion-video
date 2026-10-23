package com.stonewu.fusion.controller.media;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.media.MediaAccessTickets;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

@RestController
@RequestMapping("/api/media-resources")
@RequiredArgsConstructor
public class MediaResourceController {
    private final CanvasActorResolver actors;
    private final MediaIntakeService intake;
    private final MediaResourceQueries queries;
    private final MediaAccessTickets tickets;

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public CommonResult<MediaResource> upload(@RequestParam MultipartFile file, @RequestParam String requestId,
                                              @RequestParam(required = false) Long projectId,
                                              @RequestParam(required = false) String canvasId) throws IOException {
        var context = ResourceCommandContext.user(actors.current(), canvasId);
        try (var stream = file.getInputStream()) {
            return CommonResult.success(intake.upload(stream, file.getOriginalFilename(), file.getContentType(),
                    requestId, projectId, context));
        }
    }

    @GetMapping("/{id}")
    public CommonResult<MediaResource> get(@PathVariable String id) {
        return CommonResult.success(queries.get(id, actors.current()));
    }

    @GetMapping("/{id}/sources")
    public CommonResult<List<MediaSourceReference>> sources(@PathVariable String id) {
        return CommonResult.success(queries.sources(id, actors.current()));
    }

    @GetMapping
    public CommonResult<MediaResourceQueries.Page> list(@RequestParam(required = false) Long projectId,
                                                        @RequestParam(defaultValue = "1") int page,
                                                        @RequestParam(defaultValue = "30") int pageSize) {
        return CommonResult.success(queries.list(projectId, page, pageSize, actors.current()));
    }

    @PostMapping("/{id}/access")
    public CommonResult<MediaAccessTickets.Issued> access(@PathVariable String id,
                                                          @RequestParam(defaultValue = "false") boolean download) {
        return CommonResult.success(tickets.browser(id, actors.current(), download));
    }
}
