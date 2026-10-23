package com.stonewu.fusion.service.canvas.run.mcp;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.ai.agentscope.mcp.McpWorkflowTool;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Ports;
import io.modelcontextprotocol.spec.McpSchema;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Map;
import static com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.*;
import static org.assertj.core.api.Assertions.*;

class CanvasMcpSchemasTests {
    private final ObjectMapper mapper = new ObjectMapper();
    private final CanvasJson json = new CanvasJson(mapper);
    private final CanvasMcpSchemas schemas = new CanvasMcpSchemas();
    private final CanvasMcpProcessor processor = new CanvasMcpProcessor(json, schemas);
    @Test void validatesRequiredFieldsTypesAndNestedLocalReferencesWithoutRejectingAnIdParameter() throws Exception {
        var schema = mapper.readTree("""
                {"type":"object","properties":{"id":{"$ref":"#/$defs/id"},"count":{"type":"integer","minimum":1}},
                 "required":["id","count"],"additionalProperties":false,"$defs":{"id":{"type":"string","minLength":2}}}
                """);
        schemas.validate(schema, mapper.readTree("{\"id\":\"valid\",\"count\":2}"), "输入");
        for (String invalid : List.of("{}", "{\"id\":\"x\",\"count\":2}", "{\"id\":\"ok\",\"count\":0}", "{\"id\":\"ok\",\"count\":2,\"extra\":true}"))
            assertThatThrownBy(() -> schemas.validate(schema, mapper.readTree(invalid), "输入")).hasMessageContaining("不符合");
    }
    @Test void blocksRemoteReferencesAndUnknownDialectsButKeepsLiteralDataOutOfSchemaTraversal() throws Exception {
        for (String forbidden : List.of("{\"$ref\":\"http://127.0.0.1:1/private\"}", "{\"$schema\":\"https://example.test/schema\"}", "{\"$id\":\"file:///private\"}"))
            assertThatThrownBy(() -> schemas.compile(mapper.readTree(forbidden))).isInstanceOf(RuntimeException.class);
        schemas.compile(mapper.readTree("{\"type\":\"object\",\"default\":{\"$ref\":\"example data\",\"id\":1},\"properties\":{\"id\":{\"type\":\"integer\"}}}"));
    }
    @Test void responseMustMatchTheSelectedRepresentationAndCannotFabricateStructuredOrSuccessfulOutput() throws Exception {
        var text = definition(OutputMode.TEXT, "{\"type\":\"string\",\"minLength\":2}", Idempotency.READ_ONLY, "");
        var success = processor.response(text, new McpSchema.CallToolResult(List.of(new McpSchema.TextContent("核实后的内容")), false));
        assertThat(success.path("valid").asBoolean()).isTrue();
        assertThat(success.path("output").asText()).isEqualTo("核实后的内容");
        assertThat(processor.response(text, new McpSchema.CallToolResult(List.of(new McpSchema.TextContent("失败原因")), true)).path("valid").asBoolean()).isFalse();
        var structured = definition(OutputMode.STRUCTURED_JSON, "{\"type\":\"object\",\"required\":[\"result\"]}", Idempotency.READ_ONLY, "");
        assertThat(processor.response(structured, new McpSchema.CallToolResult(List.of(new McpSchema.TextContent("{\"result\":1}")), false)).path("valid").asBoolean()).isFalse();
        var actual = new McpSchema.CallToolResult(List.of(), false, Map.of("result", "真实结果"), Map.of());
        assertThat(processor.response(structured, actual).path("output").path("result").asText()).isEqualTo("真实结果");
    }
    @Test void injectsOnlyTheStableSubmissionIdAndRejectsAnAttemptToOverrideIt() throws Exception {
        var definition = definition(OutputMode.TEXT, "{\"type\":\"string\"}", Idempotency.KEYED, "requestKey");
        var contract = new Contract("contract", 1, "操作", 0, true, definition, "hash");
        var node = new CanvasNode("node", CanvasNodeKind.MCP_TOOL, 1, "调用", null,
                json.object(Map.of("argumentsJson", "{\"id\":\"target\"}", "inputMode", "NONE")), null, 0, 0);
        var arguments = processor.arguments(node, contract, Ports.empty(), "unique-submission");
        assertThat(arguments.path("requestKey").asText()).isEqualTo("unique-submission");
        node.config().put("argumentsJson", "{\"id\":\"target\",\"requestKey\":\"forged\"}");
        assertThatThrownBy(() -> processor.arguments(node, contract, Ports.empty(), "unique-submission")).hasMessageContaining("不能");
    }
    @Test void toolFingerprintsIgnoreJsonObjectOrderButDetectAChangedOutputContract() throws Exception {
        assertThat(McpWorkflowTool.fingerprint(mapper.readTree("{\"b\":2,\"a\":{\"y\":2,\"x\":1}}"), mapper))
                .isEqualTo(McpWorkflowTool.fingerprint(mapper.readTree("{\"a\":{\"x\":1,\"y\":2},\"b\":2}"), mapper))
                .isNotEqualTo(McpWorkflowTool.fingerprint(mapper.readTree("{\"a\":{\"x\":1,\"y\":3},\"b\":2}"), mapper));
    }
    private Definition definition(OutputMode mode, String output, Idempotency idempotency, String parameter) throws Exception {
        var schema = mapper.readTree("{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"string\"},\"requestKey\":{\"type\":\"string\"}},\"required\":[\"id\"]}");
        var tool = new McpWorkflowTool("测试", "lookup", "测试", schema, mapper.readTree(output), true, true, false, "config", "schema");
        return new Definition(tool, Effect.READ_ONLY, idempotency, parameter, mode, mapper.readTree(output), 2);
    }
}
