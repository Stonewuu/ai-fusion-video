package com.stonewu.fusion.controller.ai.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Min;
import lombok.Data;

import java.util.List;
import java.util.Map;

@Schema(description = "创建AI模型请求")
@Data
public class AiModelCreateReqVO {
    @NotBlank(message = "模型名称不能为空")
    private String name;
    @NotBlank(message = "模型标识不能为空")
    private String code;
    private String modelProtocol;
    private String capabilityPresetCode;
    @NotNull(message = "模型类型不能为空")
    private Integer modelType;
    private String icon;
    private String description;
    private Integer sort;
    private String config;
    @Min(value = 0, message = "最大并发数不能小于 0")
    private Integer maxConcurrency;
    private Boolean defaultModel;
    private Boolean supportVision;
    @NotNull(message = "多模态输入类型不能为空")
    private List<String> multimodalInputTypes;
    @NotNull(message = "多模态输入传输方式不能为空")
    private Map<String, List<String>> multimodalInputTransports;
    private Boolean supportReasoning;
    private List<String> reasoningEffortLevels;
    @Min(value = 0, message = "上下文窗口不能小于 0")
    private Integer contextWindow;
    @NotNull(message = "API配置不能为空")
    private Long apiConfigId;

    private Long comfyuiWorkflowId;
}
