package com.stonewu.fusion.controller.ai.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Schema(description = "AI模型响应")
@Data
public class AiModelRespVO {
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
    private Integer maxConcurrency;
    private Boolean defaultModel;
    private Boolean supportVision;
    private List<String> multimodalInputTypes;
    private Map<String, List<String>> multimodalInputTransports;
    private Boolean supportReasoning;
    private Integer contextWindow;
    private Long apiConfigId;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}
