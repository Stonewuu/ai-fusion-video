package com.stonewu.fusion.controller.generation;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.PageParam;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.controller.generation.vo.ImageTaskSubmitReqVO;
import com.stonewu.fusion.convert.generation.GenerationConvert;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.consumer.ImageGenerationConsumer;
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
    private final ImageGenerationConsumer imageGenerationConsumer;

    @Operation(summary = "提交生图任务")
    @PostMapping("/submit")
    public CommonResult<String> submit(@Valid @RequestBody ImageTaskSubmitReqVO reqVO) {
        ImageTask task = GenerationConvert.INSTANCE.convert(reqVO);
        task.setUserId(requireCurrentUserId());
        String taskId = imageGenerationConsumer.submitTask(task);
        return CommonResult.success(taskId);
    }

    @Operation(summary = "查询生图任务")
    @GetMapping("/{taskId}")
    public CommonResult<ImageTask> get(@PathVariable String taskId) {
        return CommonResult.success(imageGenerationService.getByTaskId(taskId));
    }

    @Operation(summary = "查询生图任务的图片条目")
    @GetMapping("/{id}/items")
    public CommonResult<List<ImageItem>> listItems(@PathVariable Long id) {
        return CommonResult.success(imageGenerationService.listItems(id));
    }

    @Operation(summary = "分页查询当前用户的生图任务")
    @GetMapping("/page")
    public CommonResult<PageResult<ImageTask>> page(PageParam pageParam) {
        Long userId = requireCurrentUserId();
        return CommonResult.success(imageGenerationService.pageByUser(userId,
                pageParam.getPageNo(), pageParam.getPageSize()));
    }
}
