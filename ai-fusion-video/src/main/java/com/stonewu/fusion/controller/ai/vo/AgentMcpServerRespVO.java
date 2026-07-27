package com.stonewu.fusion.controller.ai.vo;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public record AgentMcpServerRespVO(
        Long id,
        String name,
        String transport,
        String url,
        Map<String, String> headers,
        Map<String, String> queryParams,
        List<String> enabledTools,
        List<String> protocolVersions,
        Integer timeoutSeconds,
        Integer initializationTimeoutSeconds,
        Integer status,
        String lastTestStatus,
        String lastTestMessage,
        LocalDateTime updateTime) {
}
