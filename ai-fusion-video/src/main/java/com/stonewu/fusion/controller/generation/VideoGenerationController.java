package com.stonewu.fusion.controller.generation;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.PageParam;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.controller.generation.vo.VideoTaskSubmitReqVO;
import com.stonewu.fusion.convert.generation.GenerationConvert;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.GenerationItemViews;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.GenerationTaskViews;
import com.stonewu.fusion.service.generation.job.GenerationJobCommands;
import com.stonewu.fusion.service.generation.job.GenerationJobQueries;
import com.stonewu.fusion.service.generation.job.GenerationSubmissionService;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import static com.stonewu.fusion.security.SecurityUtils.requireCurrentUserId;

/**
 * 生视频 Controller
 */
@Tag(name = "视频生成")
@RestController
@RequestMapping("/api/generation/video")
@RequiredArgsConstructor
public class VideoGenerationController {

    private final VideoGenerationService videoGenerationService;
    private final GenerationSubmissionService submissions;
    private final GenerationJobCommands commands;
    private final GenerationJobQueries jobs;
    private final CanvasActorResolver actors;
    private final GenerationExecutionContexts executions;
    private final GenerationTaskViews views;

    @Operation(summary = "提交生视频任务")
    @PostMapping("/submit")
    public CommonResult<String> submit(@Valid @RequestBody VideoTaskSubmitReqVO reqVO) {
        var actor = actors.current();
        reqVO.getScope().require(actor);
        VideoTask task = GenerationConvert.INSTANCE.convert(reqVO);
        task.setUserId(requireCurrentUserId());
        String taskId = submissions.video(task, reqVO.getRequestId(), ResourceCommandContext.user(actor, null), null)
                .publicTaskId();
        return CommonResult.success(taskId);
    }

    @Operation(summary = "查询生视频任务")
    @GetMapping("/{taskId}")
    public CommonResult<GenerationTaskViews.View<VideoTask>> get(@PathVariable String taskId) {
        var task = videoGenerationService.getByTaskId(taskId);
        var actor = actors.current();
        executions.requireRead(GenerationKind.VIDEO, task.getId(), actor);
        return CommonResult.success(views.video(task, actor));
    }

    @Operation(summary = "查询生视频任务的视频条目")
    @GetMapping("/{id}/items")
    public CommonResult<List<VideoItem>> listItems(@PathVariable Long id) {
        executions.requireRead(GenerationKind.VIDEO, id, actors.current());
        var task = videoGenerationService.getById(id);
        return CommonResult.success(videoGenerationService.listItems(id).stream()
                .map(item -> GenerationItemViews.project(item, task.getExecutionContextJson() != null,
                        task.getStatus())).toList());
    }

    @Operation(summary = "分页查询当前用户的生视频任务")
    @GetMapping("/page")
    public CommonResult<PageResult<GenerationTaskViews.View<VideoTask>>> page(PageParam pageParam) {
        var actor = actors.current();
        return CommonResult.success(
                views.videos(videoGenerationService.pageByUser(actor, executions.readableProjects(actor),
                        pageParam.getPageNo(), pageParam.getPageSize()), actor));
    }

    @Operation(summary = "取消视频生成任务")
    @PostMapping("/{taskId}/cancel")
    public CommonResult<Boolean> cancel(@PathVariable String taskId) {
        var task = videoGenerationService.getByTaskId(taskId);
        return CommonResult.success(commands.cancel(GenerationKind.VIDEO, task.getId(), actors.current()));
    }

    @GetMapping("/{taskId}/execution")
    public CommonResult<GenerationJobQueries.View> execution(@PathVariable String taskId) {
        var task = videoGenerationService.getByTaskId(taskId);
        return CommonResult.success(jobs.task(GenerationKind.VIDEO, task.getId(), actors.current()));
    }

    @Operation(summary = "删除视频生成任务")
    @DeleteMapping("/{id}")
    public CommonResult<Boolean> deleteVideoTask(@PathVariable Long id) {
        videoGenerationService.delete(id, actors.current());
        return CommonResult.success(true);
    }
}
