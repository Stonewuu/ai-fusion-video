package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class ProjectFields {
    private static final Map<String, String> COLUMNS = Map.ofEntries(
            Map.entry("name", "name"), Map.entry("description", "description"), Map.entry("coverUrl", "cover_url"),
            Map.entry("status", "status"), Map.entry("properties", "properties"), Map.entry("artStyle", "art_style"),
            Map.entry("artStyleDescription", "art_style_description"),
            Map.entry("artStyleImagePrompt", "art_style_image_prompt"),
            Map.entry("artStyleImageUrl", "art_style_image_url"));
    private final CanvasJson json;

    public String column(String field) {
        String column = COLUMNS.get(field);
        if (column == null) {
            throw invalid("不可修改的项目字段: " + field);
        }
        return column;
    }

    public Map<String, Object> values(ObjectNode fields, boolean creating) {
        Map<String, Object> values = new LinkedHashMap<>();
        fields.fields().forEachRemaining(entry -> {
            String name = entry.getKey();
            JsonNode value = entry.getValue();
            column(name);
            if (value.isNull()) {
                if (name.equals("name") || name.equals("status")) {
                    throw invalid("项目名称和状态不能清空");
                }
                values.put(name, null);
            } else if (name.equals("status")) {
                if (!value.isIntegralNumber() || !value.canConvertToInt() || value.intValue() < 0 || value.intValue() > 3) {
                    throw invalid("项目状态无效");
                }
                values.put(name, value.intValue());
            } else if (name.equals("properties")) {
                if (!value.isObject()) {
                    throw invalid("项目扩展属性必须是 JSON 对象");
                }
                String encoded = json.write(value);
                if (encoded.getBytes(StandardCharsets.UTF_8).length > 262144) {
                    throw invalid("项目扩展属性超过 256 KiB");
                }
                values.put(name, encoded);
            } else {
                if (!value.isTextual()) {
                    throw invalid("项目字段必须为文本: " + name);
                }
                String text = value.textValue();
                int limit = switch (name) {
                    case "name" -> 256;
                    case "artStyle" -> 64;
                    case "coverUrl", "artStyleImageUrl" -> 1024;
                    default -> 65535;
                };
                if (text.length() > limit || text.getBytes(StandardCharsets.UTF_8).length > 65535) {
                    throw invalid("项目字段过长: " + name);
                }
                if (name.equals("name") && text.isBlank()) {
                    throw invalid("项目名称不能为空");
                }
                if (name.equals("coverUrl") || name.equals("artStyleImageUrl")) {
                    String lower = text.strip().toLowerCase(Locale.ROOT);
                    if (lower.startsWith("data:") || lower.startsWith("blob:")) {
                        throw invalid("项目图片必须使用已保存的媒体地址");
                    }
                }
                values.put(name, text);
            }
        });
        if (creating && !values.containsKey("name")) {
            throw invalid("创建项目必须提供名称");
        }
        return values;
    }

    private CanvasException invalid(String message) {
        return new CanvasException(CanvasErrorCode.INVALID_COMMAND, message);
    }
}
