package com.stonewu.fusion.controller.canvas;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.CanvasPlacementService;
import com.stonewu.fusion.service.canvas.CanvasPlacementService.CreateAndPlace;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceAccessPolicy;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.*;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.Objects;

@RestController
@RequestMapping("/api/canvases/{canvasId}/resource-commands")
@RequiredArgsConstructor
public class CanvasResourceController {
    private final ResourceCommandService commands;
    private final CanvasActorResolver actors;
    private final ResourceQueryService queries;
    private final CanvasDocumentRepository canvases;
    private final CanvasAccessPolicy access;
    private final CanvasPlacementService placement;
    private final ResourceAccessPolicy resourceAccess;

    public record ResourceView(ResourceHeader header, ObjectNode data, boolean canEdit) {
    }

    @PostMapping("/create-and-place")
    public CommonResult<CanvasPlacementService.Result> createAndPlace(@PathVariable String canvasId,
                                                                      @Valid @RequestBody CreateAndPlace command) {
        return CommonResult.success(placement.create(canvasId, command, actors.current()));
    }

    @PostMapping
    public CommonResult<ResourceCommandReceipt> patch(@PathVariable String canvasId,
                                                      @Valid @RequestBody ResourcePatchCommand command) {
        return CommonResult.success(commands.patch(command, ResourceCommandContext.user(actors.current(), canvasId)));
    }

    @PostMapping("/create")
    public CommonResult<ResourceCommandReceipt> create(@PathVariable String canvasId,
                                                       @Valid @RequestBody ResourceCreateCommand command) {
        return CommonResult.success(commands.execute(command, ResourceCommandContext.user(actors.current(), canvasId)));
    }

    @PostMapping("/delete")
    public CommonResult<ResourceCommandReceipt> delete(@PathVariable String canvasId,
                                                       @Valid @RequestBody ResourceDeleteCommand command) {
        return CommonResult.success(commands.execute(command, ResourceCommandContext.user(actors.current(), canvasId)));
    }

    @PostMapping("/initialize-workspace")
    public CommonResult<ResourceCommandReceipt> initialize(@PathVariable String canvasId,
                                                           @Valid @RequestBody InitializeProjectWorkspaceCommand command) {
        return CommonResult.success(commands.execute(command, ResourceCommandContext.user(actors.current(), canvasId)));
    }

    @GetMapping("/{type}/{resourceId}")
    public CommonResult<ResourceView> get(@PathVariable String canvasId, @PathVariable String type,
                                          @PathVariable long resourceId) {
        var actor = actors.current();
        var canvas = canvases.require(canvasId, false);
        var permissions = access.requireView(canvas, actor);
        CanvasValueType valueType;
        try {
            valueType = CanvasValueType.fromCode(type);
        } catch (IllegalArgumentException invalid) {
            throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "业务对象类型无效");
        }
        var resource = queries.get(valueType, resourceId, actor);
        if (!Objects.equals(resource.header().projectId(), canvas.projectId())) {
            throw new CanvasException(403, CanvasErrorCode.ACCESS_DENIED, "对象不属于当前画布项目", Map.of());
        }
        return CommonResult.success(new ResourceView(resource.header(), resource.data(),
                permissions.canEdit() && !canvas.archived() && resourceAccess.canEdit(resource.header(), actor)));
    }

    @PostMapping("/script-source")
    public CommonResult<ResourceCommandReceipt> scriptSource(@PathVariable String canvasId,
                                                             @Valid @RequestBody ScriptSourceCommand command) {
        return CommonResult.success(commands.execute(command, ResourceCommandContext.user(actors.current(), canvasId)));
    }

    @PostMapping("/script-scenes")
    public CommonResult<ResourceCommandReceipt> scriptScenes(@PathVariable String canvasId,
                                                             @Valid @RequestBody ScriptScenesCommand command) {
        return CommonResult.success(commands.execute(command, ResourceCommandContext.user(actors.current(), canvasId)));
    }

    @PostMapping("/asset-batch")
    public CommonResult<ResourceCommandReceipt> assetBatch(@PathVariable String canvasId,
                                                           @Valid @RequestBody AssetBatchCreateCommand command) {
        return CommonResult.success(commands.execute(command, ResourceCommandContext.user(actors.current(), canvasId)));
    }

    @PostMapping("/storyboard-children")
    public CommonResult<ResourceCommandReceipt> storyboardChildren(@PathVariable String canvasId,
                                                                   @Valid @RequestBody StoryboardChildrenCommand command) {
        return CommonResult.success(commands.execute(command, ResourceCommandContext.user(actors.current(), canvasId)));
    }

    @PostMapping("/storyboard-batch")
    public CommonResult<ResourceCommandReceipt> storyboardBatch(@PathVariable String canvasId,
                                                                @Valid @RequestBody StoryboardBatchPatchCommand command) {
        return CommonResult.success(commands.execute(command, ResourceCommandContext.user(actors.current(), canvasId)));
    }
}
