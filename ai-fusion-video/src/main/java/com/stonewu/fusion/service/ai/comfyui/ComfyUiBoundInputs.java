package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;

import java.util.ArrayList;
import java.util.Map;

/**
 * 画布的完整 ComfyUI 参数与端口输入，保留数值、布尔和列表类型。
 */
public final class ComfyUiBoundInputs {
    private ComfyUiBoundInputs() {
    }

    public static void overlay(Map<String, Object> defaults, ObjectNode frozen, int count) {
        if (frozen == null) {
            return;
        }
        frozen.fields().forEachRemaining(entry -> {
            if (entry.getKey().equals("seed") && entry.getValue().asLong() == -1) {
                return;
            }
            defaults.put(entry.getKey(), value(entry.getValue()));
        });
        defaults.put("count", count);
    }

    private static Object value(JsonNode value) {
        if (value.isTextual()) {
            return value.textValue();
        }
        if (value.isIntegralNumber()) {
            return value.longValue();
        }
        if (value.isFloatingPointNumber()) {
            return value.doubleValue();
        }
        if (value.isBoolean()) {
            return value.booleanValue();
        }
        if (value.isArray()) {
            var result = new ArrayList<>();
            value.forEach(item -> result.add(value(item)));
            return result;
        }
        throw new BusinessException(400, "ComfyUI 绑定输入只支持文本、数字、布尔和列表");
    }
}
