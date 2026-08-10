package com.stonewu.fusion.controller.generation.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

/**
 * 提交生图任务请求 VO
 */
@Schema(description = "提交生图任务请求")
@Data
public class ImageTaskSubmitReqVO {

    private Long projectId;

    @NotBlank(message = "提示词不能为空")
    private String prompt;

    private Long promptTemplateId;

    private String refImageUrls;

    private String ratio;

    private String resolution;

    private String aspectRatio;

    private Integer width;

    private Integer height;

    private Integer count;

    private Long modelId;

    private String category;
}
