package com.stonewu.fusion.controller.project;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.PageParam;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.config.ArtStylePresets;
import com.stonewu.fusion.controller.project.vo.ProjectCreateReqVO;
import com.stonewu.fusion.controller.project.vo.ProjectInitializeReqVO;
import com.stonewu.fusion.controller.project.vo.ProjectUpdateReqVO;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.*;
import com.stonewu.fusion.service.system.SystemConfigService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 视频项目 Controller
 */
@Tag(name = "视频项目管理")
@RestController
@RequestMapping("/api/project")
@RequiredArgsConstructor
public class ProjectController {

    private final SystemConfigService systemConfigService;
    private final ResourceCommandService resourceCommands;
    private final CanvasActorResolver canvasActors;
    private final ResourceQueryService resourceQueries;

    @Operation(summary = "项目分页")
    @GetMapping("/page")
    public CommonResult<PageResult<ObjectNode>> page(PageParam pageParam) {
        return CommonResult.success(
                resourceQueries.projectPage(canvasActors.current(), pageParam.getPageNo(), pageParam.getPageSize()));
    }

    @Operation(summary = "获取项目详情")
    @GetMapping("/{id}")
    public CommonResult<ObjectNode> get(@PathVariable Long id) {
        return CommonResult.success(resourceQueries.get(CanvasValueType.PROJECT, id, canvasActors.current()).data());
    }

    @Operation(summary = "获取项目剧本与分镜工作区概览")
    @GetMapping("/{id}/workspace-overview")
    public CommonResult<ObjectNode> getWorkspaceOverview(@PathVariable Long id) {
        return CommonResult.success(resourceQueries.workspace(id, canvasActors.current()));
    }

    @Operation(summary = "补齐项目剧本与分镜工作区")
    @PostMapping("/{id}/workspace/initialize")
    public CommonResult<ObjectNode> initializeWorkspace(@PathVariable Long id,
                                                        @Valid @RequestBody ProjectInitializeReqVO request) {
        var command = new InitializeProjectWorkspaceCommand(request.requestId(), id, request.version());
        return CommonResult.success(
                resourceCommands.execute(command, ResourceCommandContext.user(canvasActors.current(), null)).result());
    }

    @Operation(summary = "按归属查询项目列表")
    @GetMapping("/list")
    public CommonResult<List<ObjectNode>> list() {
        return CommonResult.success(resourceQueries.projects(canvasActors.current()));
    }

    @Operation(summary = "创建项目")
    @PostMapping
    public CommonResult<ObjectNode> create(@Valid @RequestBody ProjectCreateReqVO reqVO) {
        var command = new ResourceCreateCommand(reqVO.getRequestId(), CanvasValueType.PROJECT, null, reqVO.getFields());
        return CommonResult.success(
                resourceCommands.execute(command, ResourceCommandContext.user(canvasActors.current(), null)).result());
    }

    @Operation(summary = "更新项目")
    @PutMapping
    public CommonResult<ObjectNode> update(@Valid @RequestBody ProjectUpdateReqVO reqVO) {
        ResourcePatchCommand command = new ResourcePatchCommand(reqVO.getRequestId(), CanvasValueType.PROJECT,
                reqVO.getId(), reqVO.getVersion(), reqVO.getPatch());
        return CommonResult.success(
                resourceCommands.patch(command, ResourceCommandContext.user(canvasActors.current(), null)).result());
    }

    @Operation(summary = "删除项目")
    @DeleteMapping("/{id}")
    public CommonResult<Boolean> delete(@PathVariable Long id, @RequestParam Integer version,
                                        @RequestParam String requestId) {
        resourceCommands.execute(new ResourceDeleteCommand(requestId, CanvasValueType.PROJECT, id, version),
                ResourceCommandContext.user(canvasActors.current(), null));
        return CommonResult.success(true);
    }

    // ========== 画风预设 ==========

    @Operation(summary = "获取预设画风列表")
    @GetMapping("/presets/art-styles")
    public CommonResult<List<ArtStylePresets.ArtStylePreset>> listArtStylePresets() {
        List<ArtStylePresets.ArtStylePreset> presets = ArtStylePresets.getAll();
        // 填充每个预设的公网 URL
        for (ArtStylePresets.ArtStylePreset preset : presets) {
            String publicUrl = systemConfigService.getValue("art_preset_url:" + preset.getKey());
            preset.setReferenceImagePublicUrl(publicUrl);
        }
        return CommonResult.success(presets);
    }
}
