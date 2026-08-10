package com.stonewu.fusion.service.ai.agentscope.permission;

/** User-selected tool execution policy for one assistant conversation turn. */
public enum ToolExecutionMode {
    DEFAULT,
    ALWAYS_ASK,
    ALWAYS_ALLOW,
    FULL_ACCESS;

    public static ToolExecutionMode parse(String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("toolExecutionMode must not be blank");
        }
        try {
            return valueOf(value.trim());
        } catch (IllegalArgumentException invalid) {
            throw new IllegalArgumentException("Unsupported tool execution mode: " + value, invalid);
        }
    }
}
