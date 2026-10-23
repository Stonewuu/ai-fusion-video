package com.stonewu.fusion.service.ai.agentscope.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelToolManifest;
import io.agentscope.core.tool.mcp.McpClientWrapper;
import io.agentscope.core.tool.mcp.McpTool;
import io.modelcontextprotocol.spec.McpSchema;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 只公开能力定义与不可逆摘要；连接凭据留在原 MCP 服务配置中。
 */
public record McpWorkflowTool(String serverName, String toolName, String description,
                              JsonNode inputSchema, JsonNode outputSchema, boolean readOnly,
                              boolean idempotent, boolean destructive, String configurationHash,
                              String schemaHash) {
    public static McpWorkflowTool describe(String server, McpSchema.Tool tool, String configurationHash,
                                           ObjectMapper mapper) {
        var input = mapper.valueToTree(McpTool.convertMcpSchemaToParameters(tool.inputSchema(), Set.of()));
        var output = mapper.valueToTree(tool.outputSchema());
        var annotations = tool.annotations();
        boolean readOnly = annotations != null && Boolean.TRUE.equals(annotations.readOnlyHint());
        boolean idempotent = readOnly || annotations != null && Boolean.TRUE.equals(annotations.idempotentHint());
        boolean destructive = !readOnly && (annotations == null || !Boolean.FALSE.equals(
                annotations.destructiveHint()));
        String hash;
        try {
            hash = fingerprint(Map.of(
                    "name", tool.name(), "input", input, "output", output,
                    "readOnly", readOnly, "idempotent", idempotent, "destructive", destructive), mapper);
        } catch (Exception failure) {
            throw new IllegalArgumentException("MCP 工具定义无法读取", failure);
        }
        return new McpWorkflowTool(server, tool.name(), tool.description(), input, output, readOnly,
                idempotent, destructive, configurationHash, hash);
    }

    public static String fingerprint(Object value, ObjectMapper mapper) {
        return AgentKernelToolManifest.schemaSha256(canonical(mapper.valueToTree(value), mapper).toString());
    }

    private static JsonNode canonical(JsonNode value, ObjectMapper mapper) {
        if (value.isObject()) {
            var result = mapper.createObjectNode();
            value.properties().stream().sorted(Map.Entry.comparingByKey())
                    .forEach(entry -> result.set(entry.getKey(), canonical(entry.getValue(), mapper)));
            return result;
        }
        if (value.isArray()) {
            var result = mapper.createArrayNode();
            value.forEach(item -> result.add(canonical(item, mapper)));
            return result;
        }
        return value;
    }

    public static final class Opened implements AutoCloseable {
        private final McpClientWrapper client;
        private final boolean owned;
        private final McpWorkflowTool definition;
        private final AtomicBoolean submitted = new AtomicBoolean();

        public Opened(McpClientWrapper client, boolean owned, McpWorkflowTool definition) {
            this.client = client;
            this.owned = owned;
            this.definition = definition;
        }

        public McpWorkflowTool definition() {
            return definition;
        }

        public Mono<McpSchema.CallToolResult> invoke(Map<String, Object> arguments, Duration timeout) {
            return invoke(arguments, timeout, Map.of());
        }

        public Mono<McpSchema.CallToolResult> invoke(Map<String, Object> arguments, Duration timeout,
                                                     Map<String, Object> metadata) {
            return Mono.defer(() -> {
                if (!submitted.compareAndSet(false, true)) {
                    return Mono.error(new IllegalStateException("同一 MCP 请求不能再次提交"));
                }
                return client.callTool(definition.toolName(), arguments, metadata).timeout(timeout);
            });
        }

        @Override
        public void close() {
            if (owned) {
                client.close();
            }
        }
    }
}
