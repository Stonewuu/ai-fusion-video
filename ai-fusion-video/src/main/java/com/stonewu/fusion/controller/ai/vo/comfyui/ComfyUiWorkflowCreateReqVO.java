package com.stonewu.fusion.controller.ai.vo.comfyui;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class ComfyUiWorkflowCreateReqVO {

    @NotNull(message = "请选择 ComfyUI API 配置")
    private Long apiConfigId;

    @NotBlank(message = "工作流名称不能为空")
    private String name;

    @NotBlank(message = "工作流标识不能为空")
    private String code;

    @NotNull(message = "工作流模型类型不能为空")
    private Integer modelType;

    private String description;

    private Integer status;
}
