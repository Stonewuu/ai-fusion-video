package com.stonewu.fusion.service.media.lifecycle;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayDeque;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Pattern;

/**
 * 只登记明确的托管引用；索引用于保留检查，不携带或扩大读取权限。
 */
public final class MediaReferenceIndex {
    private static final String UUID = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";
    private static final Pattern URI = Pattern.compile("media://(" + UUID + ")(?![0-9a-f-])");
    private static final Set<String> IDS = Set.of("mediaId", "resourceId", "sourceMediaId", "mediaIds");

    private record Item(String field, JsonNode value) {
    }

    private MediaReferenceIndex() {
    }

    public static String extract(String payload, ObjectMapper mapper) {
        try {
            JsonNode root = mapper.reader().with(DeserializationFeature.FAIL_ON_TRAILING_TOKENS).readTree(payload);
            if (root == null) {
                throw new IllegalArgumentException("媒体引用索引缺少正文");
            }
            var queue = new ArrayDeque<Item>();
            queue.add(new Item("", root));
            var ids = new TreeSet<String>();
            while (!queue.isEmpty()) {
                var item = queue.removeFirst();
                var node = item.value();
                if (node.isTextual()) {
                    var text = node.textValue();
                    var matcher = URI.matcher(text);
                    while (matcher.find()) {
                        ids.add(matcher.group(1));
                    }
                    if (IDS.contains(item.field()) && text.matches(UUID)) {
                        ids.add(text);
                    }
                } else if (node.isObject()) {
                    if (item.field().equals("inputMediaVersions")) {
                        node.fieldNames().forEachRemaining(id -> {
                            if (id.matches(UUID)) {
                                ids.add(id);
                            }
                        });
                    }
                    node.fields().forEachRemaining(entry -> queue.add(new Item(entry.getKey(), entry.getValue())));
                } else if (node.isArray()) {
                    node.forEach(value -> queue.add(new Item(item.field(), value)));
                }
                if (ids.size() > 10000) {
                    throw new IllegalArgumentException("单份正文的媒体引用超过 10000 项");
                }
            }
            return mapper.writeValueAsString(ids.stream().map(id -> "media://" + id).toList());
        } catch (IllegalArgumentException failure) {
            throw failure;
        } catch (Exception failure) {
            throw new IllegalArgumentException("媒体引用正文不是有效 JSON", failure);
        }
    }
}
