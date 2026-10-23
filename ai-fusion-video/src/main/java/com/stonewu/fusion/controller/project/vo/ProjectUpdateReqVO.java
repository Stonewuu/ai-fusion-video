package com.stonewu.fusion.controller.project.vo;

import com.fasterxml.jackson.databind.node.ObjectNode;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.*;
import lombok.Data;

/**
 * 更新项目请求 VO
 */
@Schema(description = "更新项目请求")
@Data
public class ProjectUpdateReqVO {

    @NotNull(message = "项目ID不能为空")
    @Positive
    private Long id;

    @NotNull
    @PositiveOrZero
    private Integer version;

    @NotBlank
    @Size(max = 128)
    @Pattern(regexp = "[A-Za-z0-9._:-]+")
    private String requestId;

    @NotNull
    private ObjectNode patch;
}
