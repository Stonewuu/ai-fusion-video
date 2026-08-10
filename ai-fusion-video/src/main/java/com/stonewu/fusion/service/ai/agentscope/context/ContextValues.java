package com.stonewu.fusion.service.ai.agentscope.context;

final class ContextValues {

    private ContextValues() {
    }

    static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value.trim();
    }

    static String requireSessionComponent(String value, String name) {
        String normalized = requireText(value, name);
        if (normalized.indexOf(':') >= 0) {
            throw new IllegalArgumentException(name + " must not contain ':'");
        }
        return normalized;
    }

    static long requirePositive(long value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive");
        }
        return value;
    }

    static Long requirePositive(Long value, String name) {
        if (value == null || value <= 0) {
            throw new IllegalArgumentException(name + " must be positive");
        }
        return value;
    }

    static Integer requirePositive(Integer value, String name) {
        if (value == null || value <= 0) {
            throw new IllegalArgumentException(name + " must be positive");
        }
        return value;
    }
}
