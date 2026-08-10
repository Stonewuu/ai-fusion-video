package com.stonewu.fusion.controller.ai.vo.comfyui;

import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class ComfyUiWorkflowUpdateReqVO {

    @NotNull(message = "工作流 ID 不能为空")
    private Long id;

    private Long apiConfigId;

    private String name;

    private String code;

    private Integer modelType;

    private String description;

    private Integer status;
}
