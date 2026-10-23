package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class CanvasAgentSchema {
    private CanvasAgentSchema() {
    }

    static void validate(JsonNode value, JsonNode schema, String path) {
        for (String keyword : List.of("oneOf", "anyOf")) {
            if (!schema.has(keyword)) {
                continue;
            }
            int matched = 0;
            BusinessException specific = null;
            for (var variant : schema.path(keyword)) {
                try {
                    validate(value, variant, path);
                    matched++;
                } catch (BusinessException invalid) {
                    for (var type : variant.path("properties").path("type").path("enum")) {
                        if (type.equals(value.path("type"))) {
                            specific = invalid;
                        }
                    }
                }
            }
            if (matched == 0 || (keyword.equals("oneOf") && matched != 1)) {
                if (specific != null) {
                    throw specific;
                }
                throw new BusinessException(400, path + " 不符合已声明的参数结构");
            }
        }
        boolean valid = switch (schema.path("type").asText()) {
            case "object" -> value.isObject();
            case "array" -> value.isArray();
            case "string" -> value.isTextual();
            case "integer" -> value.isIntegralNumber() && value.canConvertToLong();
            case "number" -> value.isNumber() && Double.isFinite(value.doubleValue());
            case "boolean" -> value.isBoolean();
            case "null" -> value.isNull();
            default -> true;
        };
        if (!valid) {
            throw new BusinessException(400, path + " 的参数类型不正确");
        }
        if (schema.has("enum")) {
            boolean found = false;
            for (var choice : schema.path("enum")) {
                if (choice.equals(value)) {
                    found = true;
                }
            }
            if (!found) {
                throw new BusinessException(400, path + " 不在允许的选项中");
            }
        }
        if (value.isObject()) {
            for (var required : schema.path("required")) {
                if (!value.has(required.asText())) {
                    throw new BusinessException(400, path + " 缺少参数：" + required.asText());
                }
            }
            value.fields().forEachRemaining(entry -> {
                var field = schema.path("properties").get(entry.getKey());
                if (field != null) {
                    validate(entry.getValue(), field, path + "." + entry.getKey());
                } else if (schema.has("additionalProperties") && !schema.path("additionalProperties").asBoolean()) {
                    throw new BusinessException(400, path + " 不接受字段：" + entry.getKey());
                }
            });
        }
        if (value.isArray()) {
            if (schema.has("minItems") && value.size() < schema.path("minItems").asInt()) {
                throw new BusinessException(400, path + " 的条目数量不足");
            }
            if (schema.has("maxItems") && value.size() > schema.path("maxItems").asInt()) {
                throw new BusinessException(400, path + " 超过允许的条目数量");
            }
            for (int index = 0; index < value.size(); index++) {
                validate(value.get(index), schema.path("items"), path + "[" + index + "]");
            }
        }
    }

    static ObjectNode string(String description) {
        return JsonNodeFactory.instance.objectNode().put("type", "string").put("description", description);
    }

    static ObjectNode number(String description) {
        return JsonNodeFactory.instance.objectNode().put("type", "number").put("description", description);
    }

    static ObjectNode integer(String description) {
        return JsonNodeFactory.instance.objectNode().put("type", "integer").put("description", description);
    }

    static ObjectNode bool(String description) {
        return JsonNodeFactory.instance.objectNode().put("type", "boolean").put("description", description);
    }

    static ObjectNode object(String description) {
        return JsonNodeFactory.instance.objectNode().put("type", "object").put("description", description);
    }

    static ObjectNode array(JsonNode items, int max) {
        var result = JsonNodeFactory.instance.objectNode().put("type", "array").put("maxItems", max);
        result.set("items", items);
        return result;
    }

    static ObjectNode choice(String description, String... values) {
        var result = string(description);
        var choices = result.putArray("enum");
        for (String value : values) {
            choices.add(value);
        }
        return result;
    }

    static ObjectNode schema(Map<String, ObjectNode> properties, String... required) {
        var result = object("").put("additionalProperties", false);
        var fields = result.putObject("properties");
        properties.forEach(fields::set);
        var requirements = result.putArray("required");
        for (String name : required) {
            requirements.add(name);
        }
        return result;
    }

    static String text(ObjectNode input, String name) {
        JsonNode value = input.get(name);
        if (value == null || value.isNull()) {
            return null;
        }
        if (!value.isTextual() || value.asText().isBlank()) {
            throw new BusinessException(400, name + " 必须为非空文本");
        }
        return value.asText();
    }

    static int integer(ObjectNode input, String name, int defaultValue) {
        JsonNode value = input.get(name);
        if (value == null || value.isNull()) {
            return defaultValue;
        }
        if (!value.isIntegralNumber() || !value.canConvertToInt()) {
            throw new BusinessException(400, name + " 必须为整数");
        }
        return value.intValue();
    }

    static long version(ObjectNode input, String name) {
        var value = input.get(name);
        if (value == null || !value.isIntegralNumber() || !value.canConvertToLong() || value.longValue() < 0) {
            throw new BusinessException(400, name + " 必须为有效版本");
        }
        return value.longValue();
    }

    static List<String> strings(ObjectNode input, String name) {
        JsonNode value = input.get(name);
        if (value == null || value.isNull()) {
            return List.of();
        }
        if (!value.isArray() || value.size() > 200) {
            throw new BusinessException(400, name + " 必须为最多 200 项的列表");
        }
        List<String> result = new ArrayList<>();
        for (JsonNode entry : value) {
            if (!entry.isTextual() || entry.asText().isBlank()) {
                throw new BusinessException(400, name + " 包含无效条目");
            }
            result.add(entry.asText());
        }
        return List.copyOf(result);
    }

    static ObjectNode object(ObjectNode input, String name) {
        if (input.get(name) == null || input.get(name).isNull()) {
            return null;
        }
        if (!(input.get(name) instanceof ObjectNode value)) {
            throw new BusinessException(400, name + " 必须为对象");
        }
        return value;
    }

    static Map<String, String> stringsMap(ObjectNode input, String name) {
        var value = object(input, name);
        if (value == null) {
            return Map.of();
        }
        var result = new LinkedHashMap<String, String>();
        value.fieldNames().forEachRemaining(key -> result.put(key, text(value, key)));
        return Map.copyOf(result);
    }
}
