package com.stonewu.fusion.controller.project.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

/**
 * 创建项目请求 VO
 */
@Schema(description = "创建项目请求")
@Data
public class ProjectCreateReqVO {

    @NotBlank(message = "项目名称不能为空")
    private String name;

    private String description;

    private String coverUrl;

    private String properties;
}
