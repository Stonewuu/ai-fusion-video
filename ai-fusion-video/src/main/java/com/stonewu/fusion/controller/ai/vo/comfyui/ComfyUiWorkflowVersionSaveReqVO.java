package com.stonewu.fusion.controller.ai.vo.comfyui;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class ComfyUiWorkflowVersionSaveReqVO {

    private Long id;

    @NotNull(message = "工作流 ID 不能为空")
    private Long workflowId;

    private String uiWorkflowJson;

    @NotBlank(message = "API-format 工作流不能为空")
    private String apiWorkflowJson;

    @NotBlank(message = "输入绑定不能为空")
    private String inputBindingsJson;

    @NotBlank(message = "输出绑定不能为空")
    private String outputBindingsJson;
}
