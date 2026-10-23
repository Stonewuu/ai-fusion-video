package com.stonewu.fusion.controller.media;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.model.ResourceBindMediaCommand;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class MediaBindingController {
    private final CanvasActorResolver actors;
    private final ResourceCommandService commands;

    @PostMapping("/api/media-bindings")
    public CommonResult<ResourceCommandReceipt> bind(@Valid @RequestBody ResourceBindMediaCommand command,
                                                     @RequestParam(required = false) String canvasId) {
        return CommonResult.success(commands.execute(command, ResourceCommandContext.user(actors.current(), canvasId)));
    }
}
