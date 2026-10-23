package com.stonewu.fusion.service.generation.transport;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;

/**
 * 已收到的失败响应。与未取得响应的连接中断分开，不要求用户确认接口已经报告的失败。
 */
public final class GenerationProviderFailure extends BusinessException {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final int httpStatus;
    private final String reason;

    private GenerationProviderFailure(int status, String reason) {
        super(502, "生成失败（HTTP " + status + "）：" + reason);
        this.httpStatus = status;
        this.reason = reason;
    }

    public int httpStatus() {
        return httpStatus;
    }

    public String reason() {
        return reason;
    }

    public static GenerationProviderFailure http(int status, String body) {
        return new GenerationProviderFailure(status, reason(body));
    }

    public static GenerationProviderFailure find(Throwable failure) {
        Set<Throwable> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Throwable current = failure; current != null && seen.add(current); current = current.getCause()) {
            if (current instanceof GenerationProviderFailure known) {
                return known;
            }
        }
        return null;
    }

    private static String reason(String body) {
        String text = body == null ? "" : body.strip();
        if (text.startsWith("{") || text.startsWith("[")) {
            try {
                text = message(JSON.readTree(text));
            } catch (JsonProcessingException invalid) {
                text = "服务返回了无法解析的错误响应";
            }
        }
        if (text == null || text.isBlank()) {
            text = "服务未提供具体原因";
        }
        if (text.matches("(?is).*<(?:!doctype|html|head|body|script)\\b.*")) {
            text = "服务返回了错误页面";
        }
        text = text.replaceAll("(?i)(?:bearer|basic)\\s+[A-Za-z0-9._~+/=-]+", "[凭据已隐藏]")
                .replaceAll("(?i)((?:api[_-]?key|access[_-]?token|password|secret)\\s*[:=]\\s*)[^\\s,;]+", "$1[已隐藏]")
                .replaceAll("\\bsk-[A-Za-z0-9_-]{8,}", "[密钥已隐藏]")
                .replaceAll("[\\r\\n\\t]+", " ").strip();
        return text.length() > 500 ? text.substring(0, 500) + "…" : text;
    }

    private static String message(JsonNode node) {
        if (node == null || node.isNull()) {
            return null;
        }
        if (node.isTextual()) {
            return node.asText();
        }
        for (String field : new String[]{"error", "message", "detail", "msg", "error_message", "errorMessage"}) {
            String text = message(node.get(field));
            if (text != null && !text.isBlank()) {
                return text;
            }
        }
        return null;
    }
}
