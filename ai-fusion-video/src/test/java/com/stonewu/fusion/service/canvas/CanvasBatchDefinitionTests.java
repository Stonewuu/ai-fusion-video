package com.stonewu.fusion.service.canvas;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.capability.CanvasBatchDefinition;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasConfigValidator;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasNodeLayout;
import org.junit.jupiter.api.Test;
import java.util.Map;
import java.util.UUID;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class CanvasBatchDefinitionTests {
    private final ObjectMapper json = new ObjectMapper();
    private final CanvasCapabilityRegistry registry = new CanvasCapabilityRegistry();
    private final CanvasConfigValidator configs = new CanvasConfigValidator();
    @Test void rejectsNestedLoopsAndUnknownOperationFields() {
        rejects(config("foreach", Map.of()), "嵌套");
        rejects(config("textGeneration", Map.of("unknown", "value")), "不支持配置字段");
        rejects(config("export", Map.of()), "不支持");
    }
    @Test void rejectsMismatchedTypesAndConflictingLiteral() {
        rejects(config("textGeneration", Map.of()).put("inputType", "image"), "类型");
        rejects(config("textGeneration", Map.of("prompt", "已手填")), "固定值");
        rejects(config("textGeneration", Map.of()).put("outputPort", "not-a-port"), "输出端口");
    }
    @Test void imageInputRequiresSharedPromptAndUsesRegisteredMediaPort() {
        var config = config("textGeneration", Map.of()).put("inputType", "image").put("inputPort", "images");
        rejects(config, "提示词");
        config.withObject("operationConfig").put("prompt", "描述图片");
        assertThat(body(config).kind()).isEqualTo(CanvasNodeKind.TEXT_GENERATION);
    }
    @Test void rejectsWorstCaseResultExpansionOverTheCollectionLimit() {
        var config = config("imageGeneration", Map.of("count", 32)).put("resultType", "image").put("outputPort", "images").put("maxItems", 32);
        rejects(config, "1000");
        config.put("maxItems", 31);
        assertThat(body(config).config().path("count").asInt()).isEqualTo(32);
    }
    private ObjectNode config(String operation, Map<String, Object> parameters) {
        return json.valueToTree(Map.of("operationKind", operation, "operationConfig", parameters, "inputType", "text",
                "resultType", "text", "inputPort", "prompt", "outputPort", "text", "maxItems", 2));
    }
    private CanvasNode body(ObjectNode config) {
        return CanvasBatchDefinition.body(new CanvasNode(UUID.randomUUID().toString(), CanvasNodeKind.FOREACH, 1, "批量",
                new CanvasNodeLayout(0, 0, null, null, null, false, false), config, null, 0, 0), registry, configs);
    }
    private void rejects(ObjectNode config, String message) {
        assertThatThrownBy(() -> body(config)).isInstanceOf(CanvasException.class).hasMessageContaining(message);
    }
}
