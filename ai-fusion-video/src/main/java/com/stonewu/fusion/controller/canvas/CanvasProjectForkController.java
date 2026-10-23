package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.collaboration.CanvasProjectForkService;
import com.stonewu.fusion.service.canvas.collaboration.CanvasProjectForkSources;
import com.stonewu.fusion.service.resource.model.ProjectForkCommand;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import static com.stonewu.fusion.common.CommonResult.success;

@RestController
@RequestMapping("/api/canvases/{canvas}/project-forks")
@RequiredArgsConstructor
public class CanvasProjectForkController {
    private final CanvasProjectForkService service;
    private final CanvasActorResolver actors;

    @GetMapping("/preview")
    public CommonResult<CanvasProjectForkSources.Preview> preview(@PathVariable String canvas,
                                                                  @RequestParam long expectedRevision) {
        return success(service.preview(canvas, expectedRevision, actors.current()));
    }

    @PostMapping
    public CommonResult<ResourceCommandReceipt> copy(@PathVariable String canvas,
                                                     @Valid @RequestBody ProjectForkCommand input) {
        if (!canvas.equals(input.sourceCanvasId())) {
            throw new BusinessException(400, "副本来源与当前画布不一致");
        }
        return success(service.copy(input, actors.current()));
    }
}
