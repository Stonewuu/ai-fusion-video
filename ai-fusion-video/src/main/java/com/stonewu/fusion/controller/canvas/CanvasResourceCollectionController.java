package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.CanvasResourceCollectionService;
import com.stonewu.fusion.service.canvas.CanvasResourceCollectionService.Collection;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/canvases/{canvasId}/collections")
@RequiredArgsConstructor
public class CanvasResourceCollectionController {
    private final CanvasResourceCollectionService collections;
    private final CanvasActorResolver actors;

    @GetMapping("/{type}/{resourceId}")
    public CommonResult<Collection> get(@PathVariable String canvasId, @PathVariable String type,
                                        @PathVariable long resourceId, @RequestParam(defaultValue = "1") int depth) {
        CanvasValueType valueType;
        try {
            valueType = CanvasValueType.fromCode(type);
        } catch (IllegalArgumentException failure) {
            throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "业务对象类型无效");
        }
        return CommonResult.success(collections.get(canvasId, valueType, resourceId, depth, actors.current()));
    }
}
