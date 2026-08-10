package com.stonewu.fusion.controller.ai.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Schema(description = "文本模型连通性检测响应")
@Data
public class AiModelConnectivityRespVO {
    private Long modelId;
    private String modelName;
    private String responseText;
    private Long durationMs;
    private LocalDateTime testedAt;
}