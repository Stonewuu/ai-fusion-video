package com.stonewu.fusion.controller.project;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.PageParam;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.config.ArtStylePresets;
import com.stonewu.fusion.controller.project.vo.ProjectCreateReqVO;
import com.stonewu.fusion.controller.project.vo.ProjectUpdateReqVO;
import com.stonewu.fusion.convert.project.ProjectConvert;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.entity.project.ProjectMember;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.project.dto.ProjectWorkspaceOverview;
import com.stonewu.fusion.service.system.SystemConfigService;
import com.stonewu.fusion.security.SecurityUtils;
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

    private final ProjectService projectService;
    private final SystemConfigService systemConfigService;

    @Operation(summary = "项目分页")
    @GetMapping("/page")
    public CommonResult<PageResult<Project>> page(PageParam pageParam) {
        return CommonResult.success(projectService.page(pageParam.getPageNo(), pageParam.getPageSize()));
    }

    @Operation(summary = "获取项目详情")
    @GetMapping("/{id}")
    public CommonResult<Project> get(@PathVariable Long id) {
        return CommonResult.success(projectService.getById(id));
    }

    @Operation(summary = "获取项目剧本与分镜工作区概览")
    @GetMapping("/{id}/workspace-overview")
    public CommonResult<ProjectWorkspaceOverview> getWorkspaceOverview(@PathVariable Long id) {
        return CommonResult.success(projectService.getWorkspaceOverview(id));
    }

    @Operation(summary = "补齐项目剧本与分镜工作区")
    @PostMapping("/{id}/workspace/initialize")
    public CommonResult<ProjectWorkspaceOverview> initializeWorkspace(@PathVariable Long id) {
        Long userId = SecurityUtils.requireCurrentUserId();
        if (!projectService.canAccessProject(id, userId)) {
            throw new BusinessException("无权初始化该项目工作区");
        }
        return CommonResult.success(projectService.initializeWorkspace(id));
    }

    @Operation(summary = "按归属查询项目列表")
    @GetMapping("/list")
    public CommonResult<List<Project>> list() {
        return CommonResult.success(projectService.listAccessibleByUser(SecurityUtils.getCurrentUserId()));
    }

    @Operation(summary = "创建项目")
    @PostMapping
    public CommonResult<Project> create(@Valid @RequestBody ProjectCreateReqVO reqVO) {
        Project project = ProjectConvert.INSTANCE.convert(reqVO);
        return CommonResult.success(projectService.create(project));
    }

    @Operation(summary = "更新项目")
    @PutMapping
    public CommonResult<Project> update(@Valid @RequestBody ProjectUpdateReqVO reqVO) {
        Project project = ProjectConvert.INSTANCE.convert(reqVO);
        return CommonResult.success(projectService.update(project));
    }

    @Operation(summary = "删除项目")
    @DeleteMapping("/{id}")
    public CommonResult<Boolean> delete(@PathVariable Long id) {
        projectService.delete(id);
        return CommonResult.success(true);
    }

    // ========== 成员管理 ==========

    @Operation(summary = "获取项目成员列表")
    @GetMapping("/{projectId}/members")
    public CommonResult<List<ProjectMember>> listMembers(@PathVariable Long projectId) {
        return CommonResult.success(projectService.listMembers(projectId));
    }

    @Operation(summary = "添加项目成员")
    @PostMapping("/{projectId}/members")
    public CommonResult<ProjectMember> addMember(@PathVariable Long projectId,
                                                  @RequestParam Long userId,
                                                  @RequestParam(defaultValue = "3") Integer role) {
        return CommonResult.success(projectService.addMember(projectId, userId, role));
    }

    @Operation(summary = "移除项目成员")
    @DeleteMapping("/{projectId}/members/{userId}")
    public CommonResult<Boolean> removeMember(@PathVariable Long projectId, @PathVariable Long userId) {
        projectService.removeMember(projectId, userId);
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
