package com.stonewu.fusion.controller.asset;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.asset.vo.AssetCreateReqVO;
import com.stonewu.fusion.controller.asset.vo.AssetItemCreateReqVO;
import com.stonewu.fusion.controller.asset.vo.AssetItemUpdateReqVO;
import com.stonewu.fusion.controller.asset.vo.AssetUpdateReqVO;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.resource.AssetFields;
import com.stonewu.fusion.service.resource.AssetResourceQueryService;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.*;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET_ITEM;

@Tag(name = "资产管理")
@RestController
@RequestMapping("/api/asset")
@RequiredArgsConstructor
public class AssetController {
    private final ResourceCommandService commands;
    private final ResourceQueryService resources;
    private final AssetResourceQueryService assets;
    private final CanvasActorResolver actors;

    @GetMapping("/metadata/{assetType}")
    public CommonResult<Map<String, Object>> getMetadata(@PathVariable String assetType) {
        var fields = AssetFields.definitions(assetType);
        return CommonResult.success(Map.of("assetType", assetType, "fields", fields));
    }

    @GetMapping("/{id}")
    public CommonResult<ObjectNode> get(@PathVariable long id) {
        return CommonResult.success(resources.get(ASSET, id, actors.current()).data());
    }

    @GetMapping("/list")
    public CommonResult<List<ObjectNode>> list(@RequestParam long projectId,
                                               @RequestParam(required = false) String type,
                                               @RequestParam(required = false) String keyword) {
        return CommonResult.success(assets.list(projectId, type, keyword, false, actors.current()));
    }

    @GetMapping("/list-with-items")
    public CommonResult<List<ObjectNode>> listWithItems(@RequestParam(required = false) Long projectId) {
        return CommonResult.success(assets.list(projectId, null, null, true, actors.current()));
    }

    @GetMapping("/all")
    public CommonResult<ObjectNode> listAll(@RequestParam(required = false) Long projectId,
                                            @RequestParam(required = false) String type,
                                            @RequestParam(required = false) String keyword,
                                            @RequestParam(defaultValue = "1") int page,
                                            @RequestParam(defaultValue = "20") int size) {
        return CommonResult.success(assets.page(projectId, type, keyword, page, size, actors.current()));
    }

    @PostMapping
    public CommonResult<ObjectNode> create(@Valid @RequestBody AssetCreateReqVO request) {
        return CommonResult.success(commands.execute(
                        new ResourceCreateCommand(request.requestId(), ASSET, request.projectId(), request.fields()), actor())
                .result());
    }

    @PutMapping
    public CommonResult<ObjectNode> update(@Valid @RequestBody AssetUpdateReqVO request) {
        return CommonResult.success(commands.patch(
                new ResourcePatchCommand(request.requestId(), ASSET, request.id(), request.version(), request.patch()),
                actor()).result());
    }

    @DeleteMapping("/{id}")
    public CommonResult<Boolean> delete(@PathVariable long id, @RequestParam int version,
                                        @RequestParam String requestId) {
        commands.execute(new ResourceDeleteCommand(requestId, ASSET, id, version), actor());
        return CommonResult.success(true);
    }

    @GetMapping("/item/{id}")
    public CommonResult<ObjectNode> getItem(@PathVariable long id) {
        return CommonResult.success(resources.get(ASSET_ITEM, id, actors.current()).data());
    }

    @GetMapping("/{assetId}/items")
    public CommonResult<List<ObjectNode>> listItems(@PathVariable long assetId) {
        return CommonResult.success(assets.items(assetId, actors.current()));
    }

    @PostMapping("/item")
    public CommonResult<ObjectNode> createItem(@Valid @RequestBody AssetItemCreateReqVO request) {
        var fields = request.fields().deepCopy();
        if (fields.has("assetId") || fields.has("parentVersion")) {
            throw AssetFields.invalid("父资产必须在请求顶层指定");
        }
        fields.put("assetId", request.assetId()).put("parentVersion", request.parentVersion());
        return CommonResult.success(
                commands.execute(new ResourceCreateCommand(request.requestId(), ASSET_ITEM, null, fields), actor())
                        .result());
    }

    @PutMapping("/item")
    public CommonResult<ObjectNode> updateItem(@Valid @RequestBody AssetItemUpdateReqVO request) {
        return CommonResult.success(commands.patch(
                new ResourcePatchCommand(request.requestId(), ASSET_ITEM, request.id(), request.version(),
                        request.patch(), request.assetVersion()), actor()).result());
    }

    @DeleteMapping("/item/{id}")
    public CommonResult<Boolean> deleteItem(@PathVariable long id, @RequestParam int version,
                                            @RequestParam int parentVersion, @RequestParam String requestId) {
        commands.execute(new ResourceDeleteCommand(requestId, ASSET_ITEM, id, version, parentVersion), actor());
        return CommonResult.success(true);
    }

    @PostMapping("/batch")
    public CommonResult<ObjectNode> batch(@Valid @RequestBody AssetBatchCreateCommand command) {
        return CommonResult.success(commands.execute(command, actor()).result());
    }

    private ResourceCommandContext actor() {
        return ResourceCommandContext.user(actors.current(), null);
    }
}
