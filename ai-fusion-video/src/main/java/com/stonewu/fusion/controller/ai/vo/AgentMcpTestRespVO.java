package com.stonewu.fusion.controller.ai.vo;

import java.util.List;

public record AgentMcpTestRespVO(
        boolean success,
        String message,
        List<Tool> tools) {

    public record Tool(String name, String description, boolean readOnly) {
    }
}
