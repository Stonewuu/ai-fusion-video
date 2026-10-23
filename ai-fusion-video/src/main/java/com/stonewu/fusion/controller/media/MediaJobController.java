package com.stonewu.fusion.controller.media;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.media.job.EpisodeAudioReviewService;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobQueries;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/media-jobs")
@RequiredArgsConstructor
public class MediaJobController {
    private final CanvasActorResolver actors;
    private final MediaJobQueries jobs;
    private final EpisodeAudioReviewService audio;

    @GetMapping("/{id}")
    public CommonResult<MediaJob.View> get(@PathVariable String id) {
        return CommonResult.success(jobs.get(id, actors.current()));
    }

    @GetMapping
    public CommonResult<List<MediaJob.View>> list(@RequestParam long projectId, @RequestParam long episodeId) {
        return CommonResult.success(jobs.list(projectId, episodeId, actors.current()));
    }

    @PostMapping("/{id}/cancel")
    public CommonResult<Boolean> cancel(@PathVariable String id) {
        return CommonResult.success(jobs.cancel(id, actors.current()));
    }

    @GetMapping("/{id}/audio")
    public CommonResult<EpisodeAudioReviewService.View> audio(@PathVariable String id) {
        return CommonResult.success(audio.detail(id, actors.current()));
    }

    @PostMapping("/{id}/audio/review")
    public CommonResult<EpisodeAudioReviewService.View> review(@PathVariable String id,
                                                               @RequestBody @Valid EpisodeAudioReviewService.Review request) {
        return CommonResult.success(audio.review(id, request, actors.current()));
    }
}
