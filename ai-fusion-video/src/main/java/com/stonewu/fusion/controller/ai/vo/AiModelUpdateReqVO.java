package com.stonewu.fusion.controller.ai.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.List;
import java.util.Map;

@Schema(description = "更新AI模型请求")
@Data
public class AiModelUpdateReqVO {
    @NotNull(message = "模型ID不能为空")
    private Long id;
    private String name;
    private String code;
    private String modelProtocol;
    private String capabilityPresetCode;
    private Integer modelType;
    private String icon;
    private String description;
    private Integer sort;
    private Integer status;
    private String config;
    @Min(value = 0, message = "最大并发数不能小于 0")
    private Integer maxConcurrency;
    private Boolean defaultModel;
    private Boolean supportVision;
    private List<String> multimodalInputTypes;
    private Map<String, List<String>> multimodalInputTransports;
    private Boolean supportReasoning;
    private List<String> reasoningEffortLevels;
    @Min(value = 0, message = "上下文窗口不能小于 0")
    private Integer contextWindow;
    private Long apiConfigId;
}
