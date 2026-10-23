package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.resource.ResourceChangeFeed;
import lombok.RequiredArgsConstructor;
import org.springframework.http.CacheControl;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class ResourceChangeFeedController {
    private final CanvasActorResolver actors;
    private final ResourceChangeFeed feed;

    @GetMapping("/api/resource-events")
    public ResponseEntity<CommonResult<ResourceChangeFeed.Page>> read(
            @RequestParam(required = false) Long afterSequence,
            @RequestParam(required = false) String scope,
            @RequestParam(defaultValue = "200") int limit) {
        return ResponseEntity.ok().cacheControl(CacheControl.noStore())
                .body(CommonResult.success(feed.read(actors.current(), afterSequence, scope, limit)));
    }
}
