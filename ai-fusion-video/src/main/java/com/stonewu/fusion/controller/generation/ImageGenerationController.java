package com.stonewu.fusion.controller.generation;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.PageParam;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.controller.generation.vo.ImageTaskSubmitReqVO;
import com.stonewu.fusion.convert.generation.GenerationConvert;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.job.GenerationSubmissionService;
import com.stonewu.fusion.service.generation.job.GenerationJobCommands;
import com.stonewu.fusion.service.generation.job.GenerationJobQueries;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.GenerationItemViews;
import com.stonewu.fusion.service.generation.GenerationTaskViews;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import static com.stonewu.fusion.security.SecurityUtils.requireCurrentUserId;

/**
 * 生图 Controller
 */
@Tag(name = "图片生成")
@RestController
@RequestMapping("/api/generation/image")
@RequiredArgsConstructor
public class ImageGenerationController {

    private final ImageGenerationService imageGenerationService;
    private final GenerationSubmissionService submissions;
    private final GenerationJobCommands commands;
    private final GenerationJobQueries jobs;
    private final CanvasActorResolver actors;
    private final GenerationExecutionContexts executions;
    private final GenerationTaskViews views;

    @Operation(summary = "提交生图任务")
    @PostMapping("/submit")
    public CommonResult<String> submit(@Valid @RequestBody ImageTaskSubmitReqVO reqVO) {
        var actor = actors.current();
        reqVO.getScope().require(actor);
        ImageTask task = GenerationConvert.INSTANCE.convert(reqVO);
        task.setUserId(requireCurrentUserId());
        String taskId = submissions.image(task, reqVO.getRequestId(), ResourceCommandContext.user(actor, null), null).publicTaskId();
        return CommonResult.success(taskId);
    }

    @Operation(summary = "查询生图任务")
    @GetMapping("/{taskId}")
    public CommonResult<GenerationTaskViews.View<ImageTask>> get(@PathVariable String taskId) {
        var task = imageGenerationService.getByTaskId(taskId);
        var actor = actors.current();
        executions.requireRead(GenerationKind.IMAGE, task.getId(), actor);
        return CommonResult.success(views.image(task, actor));
    }

    @Operation(summary = "查询生图任务的图片条目")
    @GetMapping("/{id}/items")
    public CommonResult<List<ImageItem>> listItems(@PathVariable Long id) {
        executions.requireRead(GenerationKind.IMAGE, id, actors.current());
        var task = imageGenerationService.getById(id);
        return CommonResult.success(imageGenerationService.listItems(id).stream()
                .map(item -> GenerationItemViews.project(item, task.getExecutionContextJson() != null, task.getStatus())).toList());
    }

    @Operation(summary = "分页查询当前用户的生图任务")
    @GetMapping("/page")
    public CommonResult<PageResult<GenerationTaskViews.View<ImageTask>>> page(PageParam pageParam) {
        var actor = actors.current();
        return CommonResult.success(views.images(imageGenerationService.pageByUser(actor, executions.readableProjects(actor),
                pageParam.getPageNo(), pageParam.getPageSize()), actor));
    }

    @Operation(summary = "取消图片生成任务")
    @PostMapping("/{taskId}/cancel")
    public CommonResult<Boolean> cancel(@PathVariable String taskId) {
        var task = imageGenerationService.getByTaskId(taskId);
        return CommonResult.success(commands.cancel(GenerationKind.IMAGE, task.getId(), actors.current()));
    }

    @GetMapping("/{taskId}/execution")
    public CommonResult<GenerationJobQueries.View> execution(@PathVariable String taskId) {
        var task = imageGenerationService.getByTaskId(taskId);
        return CommonResult.success(jobs.task(GenerationKind.IMAGE, task.getId(), actors.current()));
    }

    @Operation(summary = "删除图片生成任务")
    @DeleteMapping("/{id}")
    public CommonResult<Boolean> deleteImageTask(@PathVariable Long id) {
        imageGenerationService.delete(id, actors.current());
        return CommonResult.success(true);
    }
}
