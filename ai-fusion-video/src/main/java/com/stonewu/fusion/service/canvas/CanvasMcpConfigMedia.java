package com.stonewu.fusion.service.canvas;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.media.MediaReference;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Set;
import java.util.function.UnaryOperator;

/**
 * MCP 的参数 JSON 是外部协议：只识别显式 media:// 引用，不猜测普通编号的含义。
 */
@Component
@RequiredArgsConstructor
public class CanvasMcpConfigMedia {
    private final CanvasJson json;

    public void collect(CanvasNode node, Set<String> ids) {
        rewrite(node, id -> {
            ids.add(id);
            return MediaReference.of(id);
        });
    }

    public ObjectNode rewrite(CanvasNode node, UnaryOperator<String> reference) {
        return config(node.kind(), node.config(), reference);
    }

    private ObjectNode config(CanvasNodeKind kind, ObjectNode source, UnaryOperator<String> reference) {
        var result = source.deepCopy();
        if (kind == CanvasNodeKind.FOREACH && result.hasNonNull("operationKind") && result.path("operationConfig")
                .isObject()) {
            var operation = CanvasNodeKind.fromCode(result.path("operationKind").asText());
            if (operation == CanvasNodeKind.FOREACH) {
                throw invalid();
            }
            result.set("operationConfig", config(operation, (ObjectNode) result.path("operationConfig"), reference));
        }
        if (kind != CanvasNodeKind.MCP_TOOL) {
            return result;
        }
        if (result.hasNonNull("argumentsJson")) {
            rewriteField(result, "argumentsJson", reference);
        }
        if (result.path("inputMode").asText().equals("JSON_OBJECT") && result.hasNonNull("input") && !result.path(
                "input").asText().isBlank()) {
            rewriteField(result, "input", reference);
        }
        return result;
    }

    private void rewriteField(ObjectNode config, String field, UnaryOperator<String> reference) {
        ObjectNode value;
        try {
            value = json.read(config.path(field).asText(), ObjectNode.class);
        } catch (RuntimeException failure) {
            throw invalid();
        }
        if (value == null) {
            throw invalid();
        }
        config.put(field, json.write(rewriteJson(value, reference)));
    }

    private JsonNode rewriteJson(JsonNode source, UnaryOperator<String> reference) {
        if (source.isTextual() && MediaReference.managed(source.asText())) {
            return TextNode.valueOf(reference.apply(MediaReference.id(source.asText())));
        }
        if (source.isObject()) {
            var result = ((ObjectNode) source).deepCopy();
            source.fields()
                    .forEachRemaining(entry -> result.set(entry.getKey(), rewriteJson(entry.getValue(), reference)));
            return result;
        }
        if (source.isArray()) {
            var result = ((ArrayNode) source).arrayNode();
            source.forEach(value -> result.add(rewriteJson(value, reference)));
            return result;
        }
        return source;
    }

    private BusinessException invalid() {
        return new BusinessException(400, "MCP 固定参数必须是有效的 JSON 对象，请先修正节点配置");
    }
}
