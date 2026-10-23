package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.stream.StreamSupport;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class CanvasAgentGraphSchemaTests {
    private final ObjectMapper json = new ObjectMapper();

    @Test
    void advertisesEveryActualCommandAndRejectsTheRealModelOpMistake() throws Exception {
        var schema = CanvasAgentGraphSchema.operation();
        var types = StreamSupport.stream(schema.path("oneOf").spliterator(), false)
                .map(value -> value.path("properties").path("type").path("enum").get(0).asText()).toList();
        assertThat(types).containsExactlyInAnyOrderElementsOf(Arrays.stream(CanvasGraphOperation.class
                .getAnnotation(JsonSubTypes.class).value()).map(JsonSubTypes.Type::name).toList());
        var invalid = json.readTree("{\"op\":\"canvas.node.create\",\"node\":{}}");
        assertThatThrownBy(() -> CanvasAgentSchema.validate(invalid, schema, "operation"))
                .isInstanceOf(BusinessException.class);
        CanvasAgentSchema.validate(json.readTree("""
                {"type":"canvas.node.create","node":{"id":"$text","kind":"text","schemaVersion":1,
                 "title":"旁白","config":{"content":"雨夜"},"configVersion":0,"layoutVersion":0,
                 "layout":{"x":0,"y":0,"width":null,"parentId":null,"locked":false,"collapsed":false}}}
                """), schema, "operation");
    }

    @Test
    void replacingAnEdgeRequiresItsIdentityInsideTheEdgeAndRejectsTopLevelDeleteFields() throws Exception {
        var valid = json.readTree("""
                {"type":"canvas.edge.replace","expectedVersion":0,"edge":{"id":"existing-edge","kind":"annotation",
                 "sourceNodeId":"source","targetNodeId":"target","version":0,"sourcePort":null,"targetPort":null,"inputIndex":null,"output":null}}
                """);
        CanvasAgentSchema.validate(valid, CanvasAgentGraphSchema.operation(), "operation");
        var misplaced = valid.deepCopy();
        ((ObjectNode) misplaced).put("edgeId", "existing-edge");
        assertThatThrownBy(() -> CanvasAgentSchema.validate(misplaced, CanvasAgentGraphSchema.operation(), "operation"))
                .isInstanceOf(BusinessException.class).hasMessageContaining("不接受字段：edgeId");
        ((ObjectNode) valid.path("edge")).remove("id");
        assertThatThrownBy(() -> CanvasAgentSchema.validate(valid, CanvasAgentGraphSchema.operation(), "operation"))
                .isInstanceOf(BusinessException.class).hasMessageContaining("缺少参数：id");
    }

    @Test
    void kindChoicesMatchTheRegistryAndRejectAnnotationAsANodeKind() {
        var schema = CanvasAgentGraphSchema.kind();
        assertThat(StreamSupport.stream(schema.path("enum").spliterator(), false).map(value -> value.asText()))
                .containsExactlyElementsOf(Arrays.stream(CanvasNodeKind.values()).map(CanvasNodeKind::code).toList());
        CanvasAgentSchema.validate(json.valueToTree("note"), schema, "kind");
        assertThatThrownBy(() -> CanvasAgentSchema.validate(json.valueToTree("annotation"), schema, "kind"))
                .isInstanceOf(BusinessException.class);
    }
}
