package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.job.MediaJob;
import jakarta.validation.Valid;
import jakarta.validation.constraints.*;

/**
 * 仅由媒体任务服务构造，客户端不能提交处理结果或执行者身份。
 */
public record ComposeWritebackCommand(@NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
                                      @Positive long episodeId, @NotBlank String jobId, @NotNull Stage stage,
                                      @Valid MediaJob.Claim claim) implements ResourceCommand {
    public enum Stage {START, COMPLETE, END}

    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.STORYBOARD_EPISODE;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.COMPOSE_WRITEBACK;
    }
}
