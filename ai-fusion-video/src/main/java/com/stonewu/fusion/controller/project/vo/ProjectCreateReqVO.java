package com.stonewu.fusion.controller.project.vo;

import com.fasterxml.jackson.databind.node.ObjectNode;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;

/**
 * 创建项目请求 VO
 */
@Schema(description = "创建项目请求")
@Data
public class ProjectCreateReqVO {

    @NotBlank
    @Size(max = 128)
    @Pattern(regexp = "[A-Za-z0-9._:-]+")
    private String requestId;

    @NotNull
    private ObjectNode fields;
}
