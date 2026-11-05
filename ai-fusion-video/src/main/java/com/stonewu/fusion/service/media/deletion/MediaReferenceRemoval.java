package com.stonewu.fusion.service.media.deletion;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.node.TextNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;

import java.util.Set;

/** 清理当前编辑内容中的明确媒体引用，不改写历史审计记录或无类型的普通标识。 */
public final class MediaReferenceRemoval {
    private static final Set<String> IDS = Set.of("resourceId", "mediaId", "sourceMediaId", "mediaIds");

    private MediaReferenceRemoval() {
    }

    public static ObjectNode patch(ObjectNode original, String id) {
        ObjectNode patch = original.objectNode();
        original.fields().forEachRemaining(field -> {
            JsonNode next = remove(field.getKey(), field.getValue(), id);
            if (!field.getValue().equals(next)) {
                patch.set(field.getKey(), next);
            }
        });
        return patch;
    }

    private static JsonNode remove(String field, JsonNode value, String id) {
        if (value.isTextual()) {
            String text = value.textValue();
            if (IDS.contains(field) && text.equals(id) || text.equals("media://" + id)) {
                return NullNode.getInstance();
            }
            return TextNode.valueOf(text.replace("media://" + id, ""));
        }
        if (value.isObject()) {
            ObjectNode result = ((ObjectNode) value).objectNode();
            value.fields().forEachRemaining(entry -> {
                if (!field.equals("inputMediaVersions") || !entry.getKey().equals(id)) {
                    JsonNode next = remove(entry.getKey(), entry.getValue(), id);
                    if (!next.isNull()) {
                        result.set(entry.getKey(), next);
                    }
                }
            });
            return result;
        }
        if (value.isArray()) {
            var result = JsonNodeFactory.instance.arrayNode();
            value.forEach(item -> {
                JsonNode next = remove(field, item, id);
                if (!next.isNull()) {
                    result.add(next);
                }
            });
            return result;
        }
        return value.deepCopy();
    }
}
