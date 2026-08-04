package com.stonewu.fusion.controller.generation;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.PageParam;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.controller.generation.vo.VideoTaskSubmitReqVO;
import com.stonewu.fusion.convert.generation.GenerationConvert;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.consumer.VideoGenerationConsumer;
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
    private final VideoGenerationConsumer videoGenerationConsumer;

    @Operation(summary = "提交生视频任务")
    @PostMapping("/submit")
    public CommonResult<String> submit(@Valid @RequestBody VideoTaskSubmitReqVO reqVO) {
        VideoTask task = GenerationConvert.INSTANCE.convert(reqVO);
        task.setUserId(requireCurrentUserId());
        String taskId = videoGenerationConsumer.submitTask(task);
        return CommonResult.success(taskId);
    }

    @Operation(summary = "查询生视频任务")
    @GetMapping("/{taskId}")
    public CommonResult<VideoTask> get(@PathVariable String taskId) {
        return CommonResult.success(videoGenerationService.getByTaskId(taskId));
    }

    @Operation(summary = "查询生视频任务的视频条目")
    @GetMapping("/{id}/items")
    public CommonResult<List<VideoItem>> listItems(@PathVariable Long id) {
        return CommonResult.success(videoGenerationService.listItems(id));
    }

    @Operation(summary = "分页查询当前用户的生视频任务")
    @GetMapping("/page")
    public CommonResult<PageResult<VideoTask>> page(PageParam pageParam) {
        Long userId = requireCurrentUserId();
        return CommonResult.success(videoGenerationService.pageByUser(userId,
                pageParam.getPageNo(), pageParam.getPageSize()));
    }

    @Operation(summary = "取消视频生成任务")
    @PostMapping("/{taskId}/cancel")
    public CommonResult<Boolean> cancel(@PathVariable String taskId) {
        return CommonResult.success(videoGenerationConsumer.cancelTask(
                taskId, requireCurrentUserId()));
    }
}
