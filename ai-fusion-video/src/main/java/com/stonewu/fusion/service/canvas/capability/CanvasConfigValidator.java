package com.stonewu.fusion.service.canvas.capability;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.Map;

@Component
public class CanvasConfigValidator {
    public void validate(CanvasNodeCapability capability, ObjectNode config) {
        validate(capability, config, true);
    }

    public void validateDraft(CanvasNodeCapability capability, ObjectNode config) {
        validate(capability, config, false);
    }

    private void validate(CanvasNodeCapability capability, ObjectNode config, boolean complete) {
        if (config == null) {
            throw invalid("节点配置必须为对象", "config");
        }
        config.fieldNames().forEachRemaining(name -> {
            if (!capability.configFields().containsKey(name)) {
                throw invalid("节点不支持配置字段: " + name, name);
            }
        });
        capability.configFields().forEach((name, field) -> {
            if (complete || config.has(name)) {
                validateField(name, field, config.get(name));
            }
        });
    }

    private void validateField(String name, CanvasConfigField field, JsonNode value) {
        if (value == null) {
            if (field.required()) {
                throw invalid("缺少节点配置字段: " + name, name);
            }
            return;
        }
        boolean validType = switch (field.type()) {
            case "string" -> value.isTextual();
            case "integer" -> value.isIntegralNumber() && value.canConvertToLong();
            case "number" -> value.isNumber() && Double.isFinite(value.doubleValue());
            case "boolean" -> value.isBoolean();
            case "object" -> value.isObject() && value.toString().length() <= field.maxLength();
            case "string_array" -> stringList(value, field);
            default -> throw new IllegalStateException("未登记的配置字段类型: " + field.type());
        };
        if (!validType) {
            throw invalid("节点配置字段类型错误: " + name, name);
        }
        if (value.isTextual()) {
            if (field.maxLength() != null && value.textValue().length() > field.maxLength()) {
                throw invalid("节点配置文字超出长度限制: " + name, name);
            }
            if (!field.enumValues().isEmpty() && !field.enumValues().contains(value.textValue())) {
                throw invalid("节点配置选项无效: " + name, name);
            }
        }
        if (value.isNumber()) {
            double number = value.doubleValue();
            if ((field.minimum() != null && number < field.minimum())
                    || (field.maximum() != null && number > field.maximum())) {
                throw invalid("节点配置数值超出范围: " + name, name);
            }
        }
    }

    private boolean stringList(JsonNode value, CanvasConfigField field) {
        if (!value.isArray() || value.size() > field.maximum()) {
            return false;
        }
        var names = new HashSet<String>();
        for (var item : value) {
            if (!item.isTextual() || item.asText().isBlank()
                    || item.asText().length() > field.maxLength() || !names.add(item.asText())) {
                return false;
            }
        }
        return true;
    }

    private CanvasException invalid(String message, String field) {
        return new CanvasException(400, CanvasErrorCode.INVALID_CONFIG, message, Map.of("field", field));
    }
}
