package com.stonewu.fusion.service.ai.text;

/**
 * 字符数不能等同于 token 数；单次读取只使用配置容量的一小部分，给历史消息与生成留出空间。
 */
public final class ToolTextBudget {
    public static final int MAX_PAGE_CHARS = 64_000;
    private static final int UNCONFIGURED_PAGE_CHARS = 8_000;

    private ToolTextBudget() {
    }

    public static int pageChars(Integer contextWindow) {
        if (contextWindow == null) {
            return UNCONFIGURED_PAGE_CHARS;
        }
        if (contextWindow <= 0) {
            throw new IllegalArgumentException("模型上下文容量必须为正数");
        }
        return Math.max(1, Math.min(MAX_PAGE_CHARS, contextWindow / 8));
    }

    public static int inlineChars(Integer contextWindow) {
        return pageChars(contextWindow) * 2;
    }
}
