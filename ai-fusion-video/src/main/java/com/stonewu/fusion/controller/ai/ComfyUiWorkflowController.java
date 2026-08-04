package com.stonewu.fusion.controller.ai;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.controller.ai.vo.comfyui.ComfyUiWorkflowCreateReqVO;
import com.stonewu.fusion.controller.ai.vo.comfyui.ComfyUiWorkflowPageReqVO;
import com.stonewu.fusion.controller.ai.vo.comfyui.ComfyUiWorkflowRespVO;
import com.stonewu.fusion.controller.ai.vo.comfyui.ComfyUiWorkflowUpdateReqVO;
import com.stonewu.fusion.controller.ai.vo.comfyui.ComfyUiWorkflowVersionRespVO;
import com.stonewu.fusion.controller.ai.vo.comfyui.ComfyUiWorkflowVersionSaveReqVO;
import com.stonewu.fusion.controller.ai.vo.comfyui.ComfyUiWorkflowValidationRespVO;
import com.stonewu.fusion.controller.ai.vo.comfyui.ComfyUiWorkflowTestReqVO;
import com.stonewu.fusion.controller.ai.vo.comfyui.ComfyUiWorkflowTestRespVO;
import com.stonewu.fusion.convert.ai.ComfyUiWorkflowConvert;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowValidationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

import static com.stonewu.fusion.common.CommonResult.success;

@Tag(name = "ComfyUI 工作流管理")
@RestController
@RequestMapping("/api/ai/comfyui/workflow")
@RequiredArgsConstructor
@PreAuthorize("hasRole('ADMIN')")
public class ComfyUiWorkflowController {

    private final ComfyUiWorkflowService workflowService;
    private final ComfyUiWorkflowValidationService validationService;

    @PostMapping("/create")
    @Operation(summary = "创建 ComfyUI 工作流")
    public CommonResult<Long> create(@Valid @RequestBody ComfyUiWorkflowCreateReqVO request) {
        return success(workflowService.createWorkflow(
                request.getApiConfigId(), request.getName(), request.getCode(),
                request.getModelType(), request.getDescription(), request.getStatus()));
    }

    @PutMapping("/update")
    @Operation(summary = "更新 ComfyUI 工作流元数据")
    public CommonResult<Boolean> update(@Valid @RequestBody ComfyUiWorkflowUpdateReqVO request) {
        workflowService.updateWorkflow(
                request.getId(), request.getApiConfigId(), request.getName(), request.getCode(),
                request.getModelType(), request.getDescription(), request.getStatus());
        return success(true);
    }

    @DeleteMapping("/delete")
    @Operation(summary = "删除 ComfyUI 工作流")
    public CommonResult<Boolean> delete(@RequestParam("id") Long id) {
        workflowService.deleteWorkflow(id);
        return success(true);
    }

    @GetMapping("/get")
    @Operation(summary = "获取 ComfyUI 工作流")
    public CommonResult<ComfyUiWorkflowRespVO> get(@RequestParam("id") Long id) {
        ComfyUiWorkflow workflow = workflowService.getById(id);
        return success(workflow == null ? null : ComfyUiWorkflowConvert.INSTANCE.convert(workflow));
    }

    @GetMapping("/page")
    @Operation(summary = "分页查询 ComfyUI 工作流")
    public CommonResult<PageResult<ComfyUiWorkflowRespVO>> page(
            @Valid ComfyUiWorkflowPageReqVO request) {
        return success(workflowService.getPage(
                        request.getName(), request.getCode(), request.getApiConfigId(),
                        request.getModelType(), request.getStatus(),
                        request.getPageNo(), request.getPageSize())
                .map(ComfyUiWorkflowConvert.INSTANCE::convert));
    }

    @GetMapping("/list")
    @Operation(summary = "查询可绑定的已发布 ComfyUI 工作流")
    public CommonResult<List<ComfyUiWorkflowRespVO>> list(
            @RequestParam(value = "apiConfigId", required = false) Long apiConfigId,
            @RequestParam(value = "modelType", required = false) Integer modelType) {
        return success(ComfyUiWorkflowConvert.INSTANCE.convertList(
                workflowService.getEnabledList(apiConfigId, modelType)));
    }

    @PostMapping("/version/create")
    @Operation(summary = "创建 ComfyUI 工作流版本")
    public CommonResult<Long> createVersion(
            @Valid @RequestBody ComfyUiWorkflowVersionSaveReqVO request) {
        return success(workflowService.createVersion(
                request.getWorkflowId(), request.getUiWorkflowJson(), request.getApiWorkflowJson(),
                request.getInputBindingsJson(), request.getOutputBindingsJson()));
    }

    @PutMapping("/version/update")
    @Operation(summary = "更新 ComfyUI 工作流草稿版本")
    public CommonResult<Boolean> updateVersion(
            @Valid @RequestBody ComfyUiWorkflowVersionSaveReqVO request) {
        if (request.getId() == null) {
            throw new BusinessException(400, "工作流版本 ID 不能为空");
        }
        workflowService.updateDraftVersion(
                request.getId(), request.getUiWorkflowJson(), request.getApiWorkflowJson(),
                request.getInputBindingsJson(), request.getOutputBindingsJson());
        return success(true);
    }

    @DeleteMapping("/version/delete")
    @Operation(summary = "删除 ComfyUI 工作流草稿版本")
    public CommonResult<Boolean> deleteVersion(@RequestParam("id") Long id) {
        workflowService.deleteDraftVersion(id);
        return success(true);
    }

    @GetMapping("/version/get")
    @Operation(summary = "获取 ComfyUI 工作流版本")
    public CommonResult<ComfyUiWorkflowVersionRespVO> getVersion(@RequestParam("id") Long id) {
        ComfyUiWorkflowVersion version = workflowService.getVersion(id);
        return success(version == null ? null : ComfyUiWorkflowConvert.INSTANCE.convertVersion(version));
    }

    @GetMapping("/versions")
    @Operation(summary = "查询 ComfyUI 工作流版本")
    public CommonResult<List<ComfyUiWorkflowVersionRespVO>> versions(
            @RequestParam("workflowId") Long workflowId) {
        return success(ComfyUiWorkflowConvert.INSTANCE.convertVersionList(
                workflowService.getVersions(workflowId)));
    }

    @PostMapping("/publish")
    @Operation(summary = "发布或回滚到已验证 ComfyUI 工作流版本")
    public CommonResult<Boolean> publish(
            @RequestParam("workflowId") Long workflowId,
            @RequestParam("versionId") Long versionId) {
        workflowService.publishVersion(workflowId, versionId);
        return success(true);
    }

    @PostMapping("/version/validate")
    @Operation(summary = "在目标 ComfyUI 实例校验工作流版本")
    public CommonResult<ComfyUiWorkflowValidationRespVO> validateVersion(
            @RequestParam("versionId") Long versionId) {
        return success(validationService.validateVersion(versionId));
    }

    @PostMapping("/version/test")
    @Operation(summary = "在目标 ComfyUI 实例试运行工作流版本")
    public CommonResult<ComfyUiWorkflowTestRespVO> testVersion(
            @Valid @RequestBody ComfyUiWorkflowTestReqVO request) {
        return success(validationService.testVersion(
                request.getVersionId(), request.getInputs()));
    }
}
