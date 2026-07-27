package com.stonewu.fusion.controller.ai.vo;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;

import java.util.List;
import java.util.Map;

public record AgentMcpServerSaveReqVO(
        Long id,
        @NotBlank String name,
        @NotBlank String transport,
        @NotBlank String url,
        Map<String, String> headers,
        Map<String, String> queryParams,
        List<String> enabledTools,
        List<String> protocolVersions,
        @Min(1) @Max(600) Integer timeoutSeconds,
        @Min(1) @Max(120) Integer initializationTimeoutSeconds,
        Integer status) {
}
