package com.stonewu.fusion.controller.storyboard;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.storyboard.vo.*;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.StoryboardFields;
import com.stonewu.fusion.service.resource.StoryboardResourceQueryService;
import com.stonewu.fusion.service.resource.model.*;
import com.stonewu.fusion.service.storyboard.VideoComposeService;
import com.stonewu.fusion.service.storyboard.dto.StoryboardStatistics;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Tag(name = "分镜管理")
@RestController
@RequestMapping("/api/storyboard")
@RequiredArgsConstructor
public class StoryboardController {
    private final ResourceCommandService commands;
    private final ResourceQueryService resources;
    private final StoryboardResourceQueryService boards;
    private final CanvasActorResolver actors;
    private final VideoComposeService videoComposeService;

    @GetMapping("/{id}")
    public CommonResult<ObjectNode> get(@PathVariable long id) {
        return read(STORYBOARD, id);
    }

    @GetMapping("/project/{projectId}")
    public CommonResult<ObjectNode> getByProject(@PathVariable long projectId) {
        return CommonResult.success(boards.project(projectId, actors.current()));
    }

    @PutMapping
    public CommonResult<ObjectNode> update(@Valid @RequestBody StoryboardPatchReqVO request) {
        return patch(STORYBOARD, request);
    }

    @PostMapping("/{id}/clearContent")
    public CommonResult<ObjectNode> clearContent(@PathVariable long id,
                                                 @Valid @RequestBody StoryboardVersionReqVO request) {
        return clear(STORYBOARD, id, request);
    }

    @GetMapping("/{storyboardId}/statistics")
    public CommonResult<StoryboardStatistics> getStatistics(@PathVariable long storyboardId) {
        return CommonResult.success(boards.statistics(storyboardId, actors.current()));
    }

    @GetMapping("/{storyboardId}/episodes")
    public CommonResult<List<ObjectNode>> listEpisodes(@PathVariable long storyboardId) {
        return list(STORYBOARD_EPISODE, STORYBOARD, storyboardId);
    }

    @GetMapping("/episode/{id}")
    public CommonResult<ObjectNode> getEpisode(@PathVariable long id) {
        return read(STORYBOARD_EPISODE, id);
    }

    @PostMapping("/episode")
    public CommonResult<ObjectNode> createEpisode(@Valid @RequestBody StoryboardChildCreateReqVO request) {
        return create(STORYBOARD_EPISODE, request);
    }

    @PutMapping("/episode")
    public CommonResult<ObjectNode> updateEpisode(@Valid @RequestBody StoryboardPatchReqVO request) {
        return patch(STORYBOARD_EPISODE, request);
    }

    @DeleteMapping("/episode/{id}")
    public CommonResult<Boolean> deleteEpisode(@PathVariable long id, @RequestParam int version,
                                               @RequestParam String requestId) {
        return delete(STORYBOARD_EPISODE, id, version, requestId);
    }

    @PutMapping("/episode/{id}/bindScriptEpisode")
    public CommonResult<ObjectNode> bindScriptEpisode(@PathVariable long id,
                                                      @Valid @RequestBody StoryboardScopedPatchReqVO request) {
        requireFields(request.patch(), Set.of("scriptEpisodeId"));
        return patch(STORYBOARD_EPISODE,
                new StoryboardPatchReqVO(id, request.version(), request.requestId(), request.patch()));
    }

    @PostMapping("/episode/{id}/clearContent")
    public CommonResult<ObjectNode> clearEpisodeContent(@PathVariable long id,
                                                        @Valid @RequestBody StoryboardVersionReqVO request) {
        return clear(STORYBOARD_EPISODE, id, request);
    }

    @PostMapping("/episode/{id}/compose-video")
    public CommonResult<String> composeEpisodeVideo(@PathVariable long id,
                                                    @Valid @RequestBody ComposeVideoReqVO request) {
        var actor = actors.current();
        resources.get(STORYBOARD_EPISODE, id, actor);
        return CommonResult.success(
                videoComposeService.submitCompose(id, request.version(), request.requestId(), actor(),
                        request.includeBgm()));
    }

    @GetMapping("/episode/{episodeId}/scenes")
    public CommonResult<List<ObjectNode>> listScenesByEpisode(@PathVariable long episodeId) {
        return list(STORYBOARD_SCENE, STORYBOARD_EPISODE, episodeId);
    }

    @GetMapping("/{storyboardId}/scenes")
    public CommonResult<List<ObjectNode>> listScenesByStoryboard(@PathVariable long storyboardId) {
        return list(STORYBOARD_SCENE, STORYBOARD, storyboardId);
    }

    @GetMapping("/scene/{id}")
    public CommonResult<ObjectNode> getScene(@PathVariable long id) {
        return read(STORYBOARD_SCENE, id);
    }

    @PostMapping("/scene")
    public CommonResult<ObjectNode> createScene(@Valid @RequestBody StoryboardChildCreateReqVO request) {
        return create(STORYBOARD_SCENE, request);
    }

    @PutMapping("/scene")
    public CommonResult<ObjectNode> updateScene(@Valid @RequestBody StoryboardPatchReqVO request) {
        return patch(STORYBOARD_SCENE, request);
    }

    @DeleteMapping("/scene/{id}")
    public CommonResult<Boolean> deleteScene(@PathVariable long id, @RequestParam int version,
                                             @RequestParam String requestId) {
        return delete(STORYBOARD_SCENE, id, version, requestId);
    }

    @GetMapping("/{storyboardId}/items")
    public CommonResult<List<ObjectNode>> listItems(@PathVariable long storyboardId) {
        return list(SHOT, STORYBOARD, storyboardId);
    }

    @GetMapping("/scene/{sceneId}/items")
    public CommonResult<List<ObjectNode>> listItemsByScene(@PathVariable long sceneId) {
        return list(SHOT, STORYBOARD_SCENE, sceneId);
    }

    @GetMapping("/item/{id}")
    public CommonResult<ObjectNode> getItem(@PathVariable long id) {
        return read(SHOT, id);
    }

    @PostMapping("/item")
    public CommonResult<ObjectNode> createItem(@Valid @RequestBody StoryboardChildCreateReqVO request) {
        return create(SHOT, request);
    }

    @PutMapping("/item")
    public CommonResult<ObjectNode> updateItem(@Valid @RequestBody StoryboardPatchReqVO request) {
        return patch(SHOT, request);
    }

    @PatchMapping("/item/{id}/assets")
    public CommonResult<ObjectNode> updateItemAssets(@PathVariable long id,
                                                     @Valid @RequestBody StoryboardScopedPatchReqVO request) {
        requireFields(request.patch(), Set.of("characterIds", "sceneAssetItemId", "propIds"));
        return patch(SHOT, new StoryboardPatchReqVO(id, request.version(), request.requestId(), request.patch()));
    }

    @PutMapping("/item/{id}/updateFrame")
    public CommonResult<ObjectNode> updateItemFrame(@PathVariable long id,
                                                    @Valid @RequestBody StoryboardScopedPatchReqVO request) {
        var input = request.patch().deepCopy();
        requireFields(input, Set.of("frameType", "imageUrl", "prompt"));
        var frame = input.remove("frameType");
        if (frame == null || !frame.isTextual() || !Set.of("first", "last").contains(frame.textValue())) {
            throw StoryboardFields.invalid("frameType 仅支持 first 或 last");
        }
        String prefix = frame.textValue() + "Frame";
        ObjectNode fields = input.objectNode();
        if (input.has("imageUrl")) {
            fields.set(prefix + "ImageUrl", input.get("imageUrl"));
        }
        if (input.has("prompt")) {
            fields.set(prefix + "Prompt", input.get("prompt"));
        }
        return patch(SHOT, new StoryboardPatchReqVO(id, request.version(), request.requestId(), fields));
    }

    @DeleteMapping("/item/{id}")
    public CommonResult<Boolean> deleteItem(@PathVariable long id, @RequestParam int version,
                                            @RequestParam String requestId) {
        return delete(SHOT, id, version, requestId);
    }

    @PostMapping("/children")
    public CommonResult<ObjectNode> children(@Valid @RequestBody StoryboardChildrenCommand command) {
        return CommonResult.success(commands.execute(command, actor()).result());
    }

    @PostMapping("/batch")
    public CommonResult<ObjectNode> batchPatch(@Valid @RequestBody StoryboardBatchPatchCommand command) {
        return CommonResult.success(commands.execute(command, actor()).result());
    }

    private CommonResult<ObjectNode> create(CanvasValueType type, StoryboardChildCreateReqVO request) {
        ObjectNode fields = request.fields().deepCopy();
        String key = StoryboardFields.parentKey(type);
        if (fields.has(key) || fields.has("parentVersion")) {
            throw StoryboardFields.invalid("父对象必须在请求顶层指定");
        }
        fields.put(key, request.parentId()).put("parentVersion", request.parentVersion());
        return CommonResult.success(
                commands.execute(new ResourceCreateCommand(request.requestId(), type, null, fields), actor()).result());
    }

    private CommonResult<ObjectNode> patch(CanvasValueType type, StoryboardPatchReqVO request) {
        return CommonResult.success(commands.patch(
                new ResourcePatchCommand(request.requestId(), type, request.id(), request.version(), request.patch()),
                actor()).result());
    }

    private CommonResult<Boolean> delete(CanvasValueType type, long id, int version, String request) {
        commands.execute(new ResourceDeleteCommand(request, type, id, version), actor());
        return CommonResult.success(true);
    }

    private CommonResult<ObjectNode> clear(CanvasValueType type, long id, StoryboardVersionReqVO request) {
        var receipt = commands.execute(new StoryboardChildrenCommand(request.requestId(), type, id, request.version(),
                StoryboardChildrenCommand.Mode.CLEAR, List.of(), List.of()), actor());
        return CommonResult.success((ObjectNode) receipt.result().get("parent"));
    }

    private CommonResult<ObjectNode> read(CanvasValueType type, long id) {
        return CommonResult.success(resources.get(type, id, actors.current()).data());
    }

    private CommonResult<List<ObjectNode>> list(CanvasValueType type, CanvasValueType parent, long id) {
        return CommonResult.success(boards.children(type, parent, id, actors.current()));
    }

    private ResourceCommandContext actor() {
        return ResourceCommandContext.user(actors.current(), null);
    }

    private void requireFields(ObjectNode fields, Set<String> allowed) {
        if (fields.isEmpty()) {
            throw StoryboardFields.invalid("修改字段不能为空");
        }
        fields.fieldNames().forEachRemaining(key -> {
            if (!allowed.contains(key)) {
                throw StoryboardFields.invalid("此操作不能修改 " + key);
            }
        });
    }
}
