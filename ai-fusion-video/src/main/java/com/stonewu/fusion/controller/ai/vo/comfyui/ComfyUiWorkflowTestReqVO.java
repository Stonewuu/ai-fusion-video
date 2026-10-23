package com.stonewu.fusion.controller.ai.vo.comfyui;

import com.stonewu.fusion.controller.generation.vo.GenerationScopeReqVO;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.util.Map;

@Data
@EqualsAndHashCode(callSuper = true)
public class ComfyUiWorkflowTestReqVO extends GenerationScopeReqVO {

    @NotBlank
    @Pattern(regexp = "[A-Za-z0-9_.:-]{1,96}")
    private String requestId;

    @NotBlank
    @Pattern(regexp = "[a-f0-9]{64}")
    private String workflowHash;

    @NotNull(message = "工作流版本 ID 不能为空")
    private Long versionId;

    @NotNull(message = "试运行输入不能为空")
    private Map<String, Object> inputs;
}
