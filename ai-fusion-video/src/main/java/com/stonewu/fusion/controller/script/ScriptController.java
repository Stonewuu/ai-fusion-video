package com.stonewu.fusion.controller.script;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.script.vo.ScriptChildCreateReqVO;
import com.stonewu.fusion.controller.script.vo.ScriptSourceReplaceReqVO;
import com.stonewu.fusion.controller.script.vo.ScriptUpdateReqVO;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.ScriptFields;
import com.stonewu.fusion.service.resource.model.*;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Tag(name = "剧本管理")
@RestController
@RequestMapping("/api/script")
@RequiredArgsConstructor
public class ScriptController {
    private final ResourceCommandService commands;
    private final ResourceQueryService queries;
    private final CanvasActorResolver actors;

    @GetMapping("/{id}")
    public CommonResult<ObjectNode> get(@PathVariable long id) {
        return get(SCRIPT, id);
    }

    @GetMapping("/project/{projectId}")
    public CommonResult<ObjectNode> getByProject(@PathVariable long projectId) {
        return CommonResult.success(queries.script(projectId, actors.current()));
    }

    @PutMapping
    public CommonResult<ObjectNode> update(@Valid @RequestBody ScriptUpdateReqVO request) {
        return patch(SCRIPT, request);
    }

    @PutMapping("/{id}/source")
    public CommonResult<ObjectNode> replaceSource(@PathVariable long id,
                                                  @Valid @RequestBody ScriptSourceReplaceReqVO request) {
        return CommonResult.success(commands.execute(
                        new ScriptSourceCommand(request.requestId(), id, request.version(), request.rawContent()), context())
                .result());
    }

    @GetMapping("/{scriptId}/episodes")
    public CommonResult<List<ObjectNode>> listEpisodes(@PathVariable long scriptId) {
        return CommonResult.success(queries.scriptChildren(SCRIPT_EPISODE, scriptId, actors.current()));
    }

    @GetMapping("/episode/{id}")
    public CommonResult<ObjectNode> getEpisode(@PathVariable long id) {
        return get(SCRIPT_EPISODE, id);
    }

    @PostMapping("/episode")
    public CommonResult<ObjectNode> createEpisode(@Valid @RequestBody ScriptChildCreateReqVO request) {
        return create(SCRIPT_EPISODE, request);
    }

    @PutMapping("/episode")
    public CommonResult<ObjectNode> updateEpisode(@Valid @RequestBody ScriptUpdateReqVO request) {
        return patch(SCRIPT_EPISODE, request);
    }

    @DeleteMapping("/episode/{id}")
    public CommonResult<Boolean> deleteEpisode(@PathVariable long id, @RequestParam int version,
                                               @RequestParam String requestId) {
        return delete(SCRIPT_EPISODE, id, version, requestId);
    }

    @GetMapping("/episode/{episodeId}/scenes")
    public CommonResult<List<ObjectNode>> listScenes(@PathVariable long episodeId) {
        return CommonResult.success(queries.scriptChildren(SCRIPT_SCENE, episodeId, actors.current()));
    }

    @GetMapping("/scene/{id}")
    public CommonResult<ObjectNode> getScene(@PathVariable long id) {
        return get(SCRIPT_SCENE, id);
    }

    @PostMapping("/scene")
    public CommonResult<ObjectNode> createScene(@Valid @RequestBody ScriptChildCreateReqVO request) {
        return create(SCRIPT_SCENE, request);
    }

    @PutMapping("/scene")
    public CommonResult<ObjectNode> updateScene(@Valid @RequestBody ScriptUpdateReqVO request) {
        return patch(SCRIPT_SCENE, request);
    }

    @DeleteMapping("/scene/{id}")
    public CommonResult<Boolean> deleteScene(@PathVariable long id, @RequestParam int version,
                                             @RequestParam String requestId) {
        return delete(SCRIPT_SCENE, id, version, requestId);
    }

    @PostMapping("/scenes/commands")
    public CommonResult<ObjectNode> scenes(@Valid @RequestBody ScriptScenesCommand command) {
        return CommonResult.success(commands.execute(command, context()).result());
    }

    private ResourceCommandContext context() {
        return ResourceCommandContext.user(actors.current(), null);
    }

    private CommonResult<ObjectNode> get(CanvasValueType type, long id) {
        return CommonResult.success(queries.get(type, id, actors.current()).data());
    }

    private CommonResult<ObjectNode> patch(CanvasValueType type, ScriptUpdateReqVO request) {
        return CommonResult.success(commands.patch(
                new ResourcePatchCommand(request.requestId(), type, request.id(), request.version(), request.patch()),
                context()).result());
    }

    private CommonResult<ObjectNode> create(CanvasValueType type, ScriptChildCreateReqVO request) {
        ObjectNode fields = request.fields().deepCopy();
        for (String key : List.of("scriptId", "episodeId", "parentVersion")) {
            if (fields.has(key)) {
                throw ScriptFields.invalid("父对象字段必须在请求顶层指定");
            }
        }
        fields.put(type == SCRIPT_EPISODE ? "scriptId" : "episodeId", request.parentId())
                .put("parentVersion", request.parentVersion());
        return CommonResult.success(
                commands.execute(new ResourceCreateCommand(request.requestId(), type, null, fields), context())
                        .result());
    }

    private CommonResult<Boolean> delete(CanvasValueType type, long id, int version, String requestId) {
        commands.execute(new ResourceDeleteCommand(requestId, type, id, version), context());
        return CommonResult.success(true);
    }
}
