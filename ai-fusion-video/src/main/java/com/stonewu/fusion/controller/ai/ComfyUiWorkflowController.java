package com.stonewu.fusion.controller.ai;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.controller.ai.vo.comfyui.*;
import com.stonewu.fusion.controller.generation.vo.GenerationScopeReqVO;
import com.stonewu.fusion.convert.ai.ComfyUiWorkflowConvert;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowValidationService;
import com.stonewu.fusion.service.ai.comfyui.test.ComfyUiTestService;
import com.stonewu.fusion.service.ai.comfyui.test.ComfyUiTestView;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

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
    private final CanvasActorResolver actors;
    private final ComfyUiTestService tests;

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
    public CommonResult<ComfyUiTestView> testVersion(
            @Valid @RequestBody ComfyUiWorkflowTestReqVO request) {
        var actor = actors.current();
        request.getScope().require(actor);
        return success(tests.submit(request.getRequestId(), request.getVersionId(), request.getWorkflowHash(),
                request.getInputs(), actor));
    }

    @GetMapping("/version/tests")
    public CommonResult<List<ComfyUiTestView>> testHistory(@RequestParam("versionId") Long versionId) {
        return success(tests.list(versionId, actors.current()));
    }

    @GetMapping("/tests/{id}")
    public CommonResult<ComfyUiTestView> testRun(@PathVariable String id) {
        return success(tests.get(id, actors.current()));
    }

    @PostMapping("/test-requests/{requestId}/withdraw")
    public CommonResult<Boolean> withdrawTest(@PathVariable String requestId,
                                              @Valid @RequestBody GenerationScopeReqVO request) {
        var actor = actors.current();
        request.getScope().require(actor);
        tests.withdraw(requestId, actor);
        return success(true);
    }

    @PostMapping("/tests/{id}/cancel")
    public CommonResult<ComfyUiTestView> cancelTest(@PathVariable String id,
                                                    @Valid @RequestBody GenerationScopeReqVO request) {
        var actor = actors.current();
        request.getScope().require(actor);
        return success(tests.cancel(id, actor));
    }

    @PostMapping("/tests/{id}/review")
    public CommonResult<ComfyUiTestView> reviewTest(@PathVariable String id, @Valid @RequestBody TestReview request) {
        var actor = actors.current();
        request.scope().require(actor);
        return success(tests.review(id, request.expectedRevision(), request.action(), request.note(), actor));
    }

    public record TestReview(@NotNull @Min(0) Long expectedRevision, @NotNull @Valid GenerationClientScope scope,
                             @NotBlank String action, @NotBlank @Size(max = 2000) String note) {
    }
}
