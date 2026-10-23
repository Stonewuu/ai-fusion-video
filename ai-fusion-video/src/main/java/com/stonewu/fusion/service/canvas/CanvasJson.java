package com.stonewu.fusion.service.canvas;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.TreeMap;

@Component
@RequiredArgsConstructor
public class CanvasJson {
    private final ObjectMapper mapper;

    public String write(Object value) {
        try {
            return mapper.writeValueAsString(value);
        } catch (JsonProcessingException failure) {
            throw new IllegalStateException("画布数据无法序列化", failure);
        }
    }

    public <T> T read(String value, Class<T> type) {
        try {
            return mapper.readValue(value, type);
        } catch (JsonProcessingException failure) {
            throw new IllegalStateException("画布持久化数据协议无效", failure);
        }
    }

    public ObjectNode object(Object value) {
        JsonNode node = mapper.valueToTree(value);
        if (!(node instanceof ObjectNode object)) {
            throw new IllegalArgumentException("事件内容必须为对象");
        }
        return object;
    }

    public String hash(Object value) {
        String canonical = write(canonical(mapper.valueToTree(value)));
        try {
            return HexFormat.of()
                    .formatHex(MessageDigest.getInstance("SHA-256").digest(canonical.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException failure) {
            throw new IllegalStateException("当前运行环境不支持 SHA-256", failure);
        }
    }

    private JsonNode canonical(JsonNode node) {
        if (node.isObject()) {
            TreeMap<String, JsonNode> ordered = new TreeMap<>();
            node.fields().forEachRemaining(entry -> ordered.put(entry.getKey(), canonical(entry.getValue())));
            ObjectNode result = mapper.createObjectNode();
            ordered.forEach(result::set);
            return result;
        }
        if (node.isArray()) {
            ArrayNode result = mapper.createArrayNode();
            node.forEach(item -> result.add(canonical(item)));
            return result;
        }
        return node;
    }
}
