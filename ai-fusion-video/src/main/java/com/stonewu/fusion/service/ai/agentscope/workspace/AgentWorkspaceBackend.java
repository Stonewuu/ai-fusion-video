package com.stonewu.fusion.service.ai.agentscope.workspace;

import java.util.Set;

public final class AgentWorkspaceBackend {

    public static final String DATABASE = "database";
    public static final String LOCAL = "local";
    public static final String OBJECT_STORAGE = "object_storage";
    private static final Set<String> SUPPORTED = Set.of(DATABASE, LOCAL, OBJECT_STORAGE);

    private AgentWorkspaceBackend() {
    }

    public static String requireSupported(String value) {
        String normalized = value == null ? null : value.trim().toLowerCase();
        if (!SUPPORTED.contains(normalized)) {
            throw new IllegalArgumentException("不支持的智能体工作空间存储类型: " + value);
        }
        return normalized;
    }
}
