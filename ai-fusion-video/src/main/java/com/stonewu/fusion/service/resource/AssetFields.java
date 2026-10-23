package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.asset.AssetMetadataRegistry;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.MediaReference;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.*;

@Component
@RequiredArgsConstructor
public class AssetFields {
    private final CanvasJson json;
    private static final Set<String> ASSET = Set.of("type", "name", "description", "coverUrl", "properties", "tags",
            "sourceType", "aiPrompt", "status");
    private static final Set<String> ITEM = Set.of("itemType", "name", "imageUrl", "thumbnailUrl", "properties",
            "sortOrder", "sourceType", "aiPrompt");
    private static final Set<String> TYPES = Set.of("character", "scene", "prop", "vehicle", "building", "costume",
            "effect", "material", "image");
    private static final Set<String> INTERNAL = Set.of("userId", "projectId", "ownerType", "ownerId", "assetId",
            "deleted");

    public Map<String, Object> values(CanvasValueType type, ObjectNode input, boolean creating) {
        Set<String> allowed = type == CanvasValueType.ASSET ? ASSET : ITEM;
        if (type != CanvasValueType.ASSET && type != CanvasValueType.ASSET_ITEM) {
            throw invalid("不是资产类型");
        }
        Map<String, Object> values = new LinkedHashMap<>();
        input.fields().forEachRemaining(entry -> {
            String field = entry.getKey();
            JsonNode node = entry.getValue();
            if (!allowed.contains(field)) {
                throw invalid("不允许修改资产字段：" + field);
            }
            if (node.isNull()) {
                if (Set.of("type", "itemType", "sourceType", "sortOrder", "status")
                        .contains(field) || (type == CanvasValueType.ASSET && field.equals("name"))) {
                    throw invalid(field + " 不能清空");
                }
                values.put(field, null);
            } else if (field.equals("properties")) {
                if (!node.isObject() || node.size() > 200 || json.write(node)
                        .getBytes(StandardCharsets.UTF_8).length > 262_144) {
                    throw invalid("properties 必须是大小受限的 JSON 对象");
                }
                for (String media : Set.of("sourceMediaUrl", "videoUrl", "audioUrl", "thumbnailUrl")) {
                    if (node.hasNonNull(media)) {
                        mediaUrl(node.get(media), media);
                    }
                }
                values.put(field, json.write(node));
            } else if (field.equals("tags")) {
                if (!node.isArray() || node.size() > 100) {
                    throw invalid("tags 必须为最多 100 项的数组");
                }
                for (JsonNode tag : node) {
                    text(tag, 128, "tag");
                }
                values.put(field, json.write(node));
            } else if (Set.of("sourceType", "status", "sortOrder").contains(field)) {
                int min = field.equals("sourceType") ? 1 : 0;
                int max = field.equals("status") ? 1 : field.equals("sourceType") ? 2 : 1_000_000;
                if (!node.isIntegralNumber() || !node.canConvertToInt() || node.intValue() < min || node.intValue() > max) {
                    throw invalid(field + " 整数范围无效");
                }
                values.put(field, node.intValue());
            } else {
                int length = field.equals("type") || field.equals("itemType") ? 32 : field.equals(
                        "name") ? 128 : 65_535;
                String value = field.endsWith("Url") ? mediaUrl(node, field) : text(node, length, field);
                if (field.equals("type") && !TYPES.contains(value)) {
                    throw invalid("不支持的资产类型");
                }
                if (field.equals("itemType") && !value.matches("[a-z][a-z0-9_]{0,31}")) {
                    throw invalid("子资产类型无效");
                }
                if (field.equals("name") && type == CanvasValueType.ASSET && value.isBlank()) {
                    throw invalid("资产名称不能为空");
                }
                values.put(field, value);
            }
        });
        if (creating && type == CanvasValueType.ASSET && (!values.containsKey("type") || !values.containsKey("name"))) {
            throw invalid("创建资产必须提供 type 和 name");
        }
        return values;
    }

    /**
     * 扩展属性允许保存创作草稿；已填写的标准选择字段必须使用登记的选项值。
     */
    public void validateProperties(String type, JsonNode properties) {
        if (type == null || !TYPES.contains(type)) {
            throw invalid("不支持的资产类型");
        }
        if (properties == null || properties.isNull()) {
            return;
        }
        var definitions = definitions(type);
        for (var field : definitions) {
            JsonNode value = properties.get(field.key());
            if (value == null || value.isNull()) {
                continue;
            }
            String text = text(value, 65_535, field.key());
            if (field.type().equals("select") && field.options().stream()
                    .noneMatch(option -> option.value().equals(text))) {
                throw invalid("属性 " + field.key() + " 不属于可用选项");
            }
        }
    }

    public static List<AssetMetadataRegistry.FieldDef> definitions(String type) {
        if (type == null) {
            throw invalid("资产类型不能为空");
        }
        return switch (type) {
            case "character", "scene", "prop" ->
                    Objects.requireNonNull(AssetMetadataRegistry.getFields(type), "资产标准属性未登记");
            case "vehicle", "building", "costume", "effect", "material", "image" ->
                    List.<AssetMetadataRegistry.FieldDef>of();
            default -> throw invalid("资产属性协议未登记");
        };
    }

    private String text(JsonNode node, int limit, String field) {
        if (!node.isTextual() || node.textValue().codePointCount(0, node.textValue().length()) > limit
                || node.textValue().getBytes(StandardCharsets.UTF_8).length > 65_535) {
            throw invalid(field + " 文本格式或长度无效");
        }
        return node.textValue();
    }

    private String mediaUrl(JsonNode node, String field) {
        String value = text(node, 1024, field);
        if (MediaReference.managed(value)) {
            MediaReference.id(value);
            return value;
        }
        if (value.isBlank()) {
            return value;
        }
        try {
            URI uri = new URI(value);
            boolean local = value.startsWith("/") && !value.startsWith("//") && uri.getScheme() == null;
            boolean remote = ("https".equalsIgnoreCase(uri.getScheme()) || "http".equalsIgnoreCase(
                    uri.getScheme())) && uri.getHost() != null && uri.getUserInfo() == null;
            if ((!local && !remote) || value.chars().anyMatch(Character::isISOControl)) {
                throw invalid(field + " 必须为已上传路径或 HTTP(S) 地址");
            }
            return value;
        } catch (URISyntaxException failure) {
            throw invalid(field + " 地址格式无效");
        }
    }

    public static String column(String name) {
        if (!ASSET.contains(name) && !ITEM.contains(name) && !INTERNAL.contains(name)) {
            throw invalid("资产字段未登记");
        }
        return name.replaceAll("([A-Z])", "_$1").toLowerCase(Locale.ROOT);
    }

    public static CanvasException invalid(String message) {
        return new CanvasException(CanvasErrorCode.INVALID_COMMAND, message);
    }
}
