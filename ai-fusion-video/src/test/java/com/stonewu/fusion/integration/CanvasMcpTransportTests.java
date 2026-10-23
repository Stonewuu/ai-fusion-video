package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentScopeMcpRegistry;
import org.junit.jupiter.api.Test;
import java.time.Duration;
import java.util.Set;
import java.util.Map;
import static org.assertj.core.api.Assertions.*;

class CanvasMcpTransportTests {
    @Test void actualRegistryInitializesBeforeDiscoveryAndRefreshesRemoteSchemas() {
        try (var remote = new CanvasMcpTestServer()) {
            var registry = registry(remote);
            try {
                assertThat(registry.catalogForAgent("ai_assistant_agent", 1L)).isEmpty();
                assertThat(registry.catalogForAgent("canvas_assistant", 1L)).singleElement().satisfies(tool -> assertThat(tool.toolName()).isEqualTo("inspect_text"));
                var initial = registry.workflowTools("canvas_assistant", 1L).getFirst();
                remote.reviseDefinition();
                var changed = registry.workflowTools("canvas_assistant", 1L).getFirst();
                assertThat(changed.schemaHash()).isNotEqualTo(initial.schemaHash());
                assertThat(changed.inputSchema().path("properties").has("language")).isTrue();
                var scenario = remote.scenario("真实注册", "success");
                try (var tool = registry.openWorkflowTool("canvas_assistant", 1L, "inspect_text")) {
                    assertThat(tool.definition().schemaHash()).isEqualTo(changed.schemaHash());
                    assertThat(tool.invoke(Map.of("text", "真实注册"), Duration.ofSeconds(3)).block(Duration.ofSeconds(4)).structuredContent().toString()).contains("真实注册");
                    assertThatThrownBy(() -> tool.invoke(Map.of("text", "真实注册"), Duration.ofSeconds(3)).block()).hasMessageContaining("不能再次提交");
                }
                assertThat(scenario.calls()).hasValue(1);
            } finally { registry.destroy(); }
        }
    }
    @Test void actualRegistryHttpClientDoesNotRepeatAToolPostAfterLostReply() {
        try (var remote = new CanvasMcpTestServer()) {
            var registry = registry(remote);
            try {
                var scenario = remote.scenario("断连", "drop");
                try (var tool = registry.openWorkflowTool("canvas_assistant", 1L, "inspect_text")) {
                    assertThatThrownBy(() -> tool.invoke(Map.of("text", "断连"), Duration.ofSeconds(3)).block(Duration.ofSeconds(4))).isInstanceOf(RuntimeException.class);
                }
                assertThat(scenario.calls()).hasValue(1);
            } finally { registry.destroy(); }
        }
    }
    private AgentScopeMcpRegistry registry(CanvasMcpTestServer remote) {
        var properties = new AgentScopeV2Properties();
        properties.getMcp().setEnabled(true);
        var server = new AgentScopeV2Properties.McpServer();
        server.setTransport("http"); server.setUrl(remote.url()); server.setAgentTypes(Set.of("canvas_assistant"));
        server.setTimeout(Duration.ofSeconds(5)); server.setInitializationTimeout(Duration.ofSeconds(5));
        properties.getMcp().getServers().put("本机 MCP", server);
        return new AgentScopeMcpRegistry(properties, new ObjectMapper());
    }
}
