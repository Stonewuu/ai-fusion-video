package com.stonewu.fusion.controller.ai.vo.comfyui;

import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.Map;

@Data
public class ComfyUiWorkflowTestReqVO {

    @NotNull(message = "工作流版本 ID 不能为空")
    private Long versionId;

    @NotNull(message = "试运行输入不能为空")
    private Map<String, Object> inputs;
}
