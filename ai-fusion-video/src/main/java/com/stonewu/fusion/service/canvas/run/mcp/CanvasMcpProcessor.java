package com.stonewu.fusion.service.canvas.run.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Ports;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import io.modelcontextprotocol.spec.McpSchema;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.*;

@Component
@RequiredArgsConstructor
public class CanvasMcpProcessor {
    private final CanvasJson json;
    private final CanvasMcpSchemas schemas;

    public ObjectNode arguments(CanvasNode node, Contract contract, Ports inputs, String requestId) {
        ObjectNode arguments;
        try {
            arguments = json.read(node.config().path("argumentsJson").asText("{}"), ObjectNode.class);
        } catch (RuntimeException failure) {
            throw CanvasRunException.invalid("MCP 参数必须是有效的 JSON 对象");
        }
        if (arguments == null) {
            throw CanvasRunException.invalid("MCP 参数不能为 null");
        }
        String mode = node.config().path("inputMode").asText("NONE");
        var values = inputs.at("input");
        if (mode.equals("NONE")) {
            if (!values.isEmpty()) {
                throw CanvasRunException.invalid("请指定已连接输入如何传给 MCP 工具");
            }
        } else {
            if (values.size() != 1 || values.getFirst().text() == null) {
                throw CanvasRunException.invalid("此 MCP 操作需要一个明确的文本输入");
            }
            String text = values.getFirst().text();
            if (mode.equals("TEXT_FIELD")) {
                String parameter = node.config().path("inputParameter").asText();
                if (parameter.isBlank() || !contract.definition().tool().inputSchema().path("properties")
                        .path(parameter).path("type").asText().equals("string")) {
                    throw CanvasRunException.invalid("请选择工具已声明的字符串参数来接收上游文本");
                }
                arguments.put(parameter, text);
            } else if (mode.equals("JSON_OBJECT")) {
                if (!arguments.isEmpty()) {
                    throw CanvasRunException.invalid("采用上游完整 JSON 参数时，请清空固定参数，避免隐式覆盖");
                }
                try {
                    arguments = json.read(text, ObjectNode.class);
                } catch (RuntimeException failure) {
                    throw CanvasRunException.invalid("上游文本不是 MCP 所需的 JSON 参数对象");
                }
            } else {
                throw CanvasRunException.invalid("MCP 输入方式未登记");
            }
        }
        var definition = contract.definition();
        if (arguments == null || json.write(arguments).getBytes(StandardCharsets.UTF_8).length > 128 * 1024) {
            throw CanvasRunException.invalid("MCP 参数为空或超过 128 KiB");
        }
        if (definition.idempotency() == Idempotency.KEYED) {
            if (arguments.has(definition.idempotencyParameter())) {
                throw CanvasRunException.invalid("MCP 幂等键由系统填入，不能从节点或上游覆盖");
            }
            arguments.put(definition.idempotencyParameter(), requestId);
        }
        schemas.validate(definition.tool().inputSchema(), arguments, "调用参数");
        return arguments;
    }

    public ObjectNode response(Definition definition, McpSchema.CallToolResult received) {
        if (received == null) {
            throw new IllegalStateException("MCP 调用没有返回响应");
        }
        var raw = json.object(received);
        String rawHash = json.hash(raw);
        if (json.write(raw).getBytes(StandardCharsets.UTF_8).length > 4 * 1024 * 1024) {
            return json.object(
                    Map.of("valid", false, "message", "原 MCP 调用已返回，但响应超过 4 MiB，未发布结果", "responseHash",
                            rawHash));
        }
        var record = json.object(Map.of("valid", false, "response", raw, "responseHash", rawHash));
        try {
            if (Boolean.TRUE.equals(received.isError())) {
                throw CanvasRunException.invalid("MCP 工具返回执行错误，请查看原响应");
            }
            JsonNode output;
            if (definition.outputMode() == OutputMode.STRUCTURED_JSON) {
                if (received.structuredContent() == null) {
                    throw CanvasRunException.invalid("MCP 未返回契约要求的结构化结果");
                }
                output = json.object(received.structuredContent());
                schemas.validate(definition.tool().outputSchema(), output, "远端结构化结果");
            } else {
                if (received.content() == null || received.content().isEmpty()
                        || received.content().stream().anyMatch(item -> !(item instanceof McpSchema.TextContent))) {
                    throw CanvasRunException.invalid("MCP 返回内容与文本契约不一致，原响应已保留");
                }
                String text = received.content().stream().map(McpSchema.TextContent.class::cast)
                        .map(McpSchema.TextContent::text)
                        .reduce((left, right) -> left + "\n" + right).orElseThrow();
                output = definition.outputMode() == OutputMode.TEXT ? json.object(Map.of("text", text))
                        .path("text") : json.read(text, JsonNode.class);
            }
            schemas.validate(definition.outputSchema(), output, "工具输出");
            record.put("valid", true).set("output", output);
        } catch (RuntimeException failure) {
            record.put("message",
                    failure instanceof CanvasRunException ? failure.getMessage() : "MCP 响应不符合约定的输出格式，原响应已保留");
        }
        return record;
    }
}
