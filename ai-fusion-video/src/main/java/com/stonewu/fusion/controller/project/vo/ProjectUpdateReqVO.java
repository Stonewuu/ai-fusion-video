package com.stonewu.fusion.controller.project.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 更新项目请求 VO
 */
@Schema(description = "更新项目请求")
@Data
public class ProjectUpdateReqVO {

    @NotNull(message = "项目ID不能为空")
    private Long id;

    private String name;

    private String description;

    private String coverUrl;

    private Integer status;

    private String properties;

    private String artStyle;

    private String artStyleDescription;

    private String artStyleImagePrompt;

    private String artStyleImageUrl;
}
