package com.stonewu.fusion.service.ai.agentscope.mcp;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import io.agentscope.core.tool.Toolkit;
import io.agentscope.core.tool.mcp.McpClientWrapper;
import io.modelcontextprotocol.spec.McpSchema;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class AgentScopeMcpRegistryTests {

    @Test
    void discoversFiltersAndRegistersMcpToolsForAllowedAgent() {
        AgentScopeV2Properties properties = new AgentScopeV2Properties();
        properties.getMcp().setEnabled(true);
        AgentScopeV2Properties.McpServer server = new AgentScopeV2Properties.McpServer();
        server.setTransport("http");
        server.setUrl("https://mcp.example.test");
        server.setEnabledTools(List.of("search_assets"));
        server.setAgentTypes(Set.of(AgentKernelSpecFactory.DEFAULT_AGENT_KEY));
        properties.getMcp().setServers(Map.of("assets", server));
        FakeMcpClient client = new FakeMcpClient(
                "assets", List.of(tool("search_assets", true), tool("delete_asset", false)));

        AgentScopeMcpRegistry registry = new AgentScopeMcpRegistry(
                properties, new ObjectMapper(), (name, ignored) -> client);
        try {
            assertThat(registry.manifestsForAgent(AgentKernelSpecFactory.DEFAULT_AGENT_KEY))
                    .singleElement()
                    .satisfies(manifest -> {
                        assertThat(manifest.toolName()).isEqualTo("search_assets");
                        assertThat(manifest.readOnly()).isTrue();
                        assertThat(manifest.concurrencySafe()).isFalse();
                    });
            assertThat(registry.manifestsForAgent("story_to_script")).isEmpty();

            Toolkit toolkit = new Toolkit();
            registry.register(
                    AgentKernelSpecFactory.DEFAULT_AGENT_KEY,
                    Set.of("search_assets"),
                    toolkit);

            assertThat(toolkit.getToolNames()).containsExactly("search_assets");
            assertThat(toolkit.getTool("search_assets").isReadOnly()).isTrue();
        } finally {
            registry.destroy();
        }
        assertThat(client.closed).isTrue();
    }

    private static McpSchema.Tool tool(String name, boolean readOnly) {
        return McpSchema.Tool.builder()
                .name(name)
                .description(name + " description")
                .inputSchema(new McpSchema.JsonSchema(
                        "object",
                        Map.of("query", Map.of("type", "string")),
                        List.of("query"),
                        false,
                        null,
                        null))
                .annotations(new McpSchema.ToolAnnotations(
                        null, readOnly, false, true, false, false))
                .build();
    }

    private static final class FakeMcpClient extends McpClientWrapper {
        private final List<McpSchema.Tool> tools;
        private boolean closed;

        private FakeMcpClient(String name, List<McpSchema.Tool> tools) {
            super(name);
            this.tools = List.copyOf(tools);
        }

        @Override
        public Mono<Void> initialize() {
            initialized = true;
            return Mono.empty();
        }

        @Override
        public Mono<List<McpSchema.Tool>> listTools() {
            return Mono.just(tools);
        }

        @Override
        public Mono<McpSchema.CallToolResult> callTool(
                String name,
                Map<String, Object> arguments) {
            return Mono.error(new UnsupportedOperationException("not needed by this test"));
        }

        @Override
        public Mono<McpSchema.CallToolResult> callTool(
                String name,
                Map<String, Object> arguments,
                Map<String, Object> metadata) {
            return Mono.error(new UnsupportedOperationException("not needed by this test"));
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
