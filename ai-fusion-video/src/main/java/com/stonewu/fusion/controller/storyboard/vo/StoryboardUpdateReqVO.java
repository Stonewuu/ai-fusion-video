package com.stonewu.fusion.controller.storyboard.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 更新分镜请求 VO
 */
@Schema(description = "更新分镜请求")
@Data
public class StoryboardUpdateReqVO {

    @NotNull(message = "分镜ID不能为空")
    private Long id;

    private String description;

    private String customColumns;

    private Integer totalDuration;
}
