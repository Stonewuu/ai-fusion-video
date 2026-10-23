package com.stonewu.fusion.controller.ai.vo;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class ComfyUiCapabilityReqVO {
    @NotNull
    private Long apiConfigId;
    @NotNull
    private Long comfyuiWorkflowId;
    @NotNull
    @Min(2)
    @Max(3)
    private Integer modelType;
    private String config;
    private String capabilityPresetCode;
}
