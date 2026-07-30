package com.stonewu.fusion.service.ai.agentscope.permission;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolPermissionRisk;
import com.stonewu.fusion.service.ai.agentscope.AgentScopeToolAdapter;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.run.RunLeaseGuard;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.permission.PermissionContextState;
import io.agentscope.core.permission.PermissionMode;
import io.agentscope.core.tool.ToolBase;
import io.agentscope.core.tool.ToolCallParam;
import io.agentscope.core.tool.Toolkit;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;

class AgentToolPermissionPolicyTests {

    @Test
    void executionModeMustBeExplicitAndSupported() {
        assertThat(ToolExecutionMode.parse("ALWAYS_ALLOW"))
                .isEqualTo(ToolExecutionMode.ALWAYS_ALLOW);
        assertThatThrownBy(() -> ToolExecutionMode.parse(null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("toolExecutionMode must not be blank");
        assertThatThrownBy(() -> ToolExecutionMode.parse("automatic"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Unsupported tool execution mode");
    }

    @Test
    void defaultAllowsReadsAndAsksForEdits() {
        Toolkit toolkit = new Toolkit();
        toolkit.registerAgentTool(new StubTool("read_script", true));
        toolkit.registerAgentTool(new StubTool("update_script", false));

        PermissionContextState context = AgentToolPermissionPolicy.contextFor(
                toolkit, ToolExecutionMode.DEFAULT);

        assertThat(context.getMode()).isEqualTo(PermissionMode.DEFAULT);
        assertThat(context.getAllowRules()).containsKey("read_script");
        assertThat(context.getAskRules()).containsKey("update_script");
    }

    @Test
    void alwaysAskGatesEveryToolAndFullAccessUsesBypassWithoutAskRules() {
        Toolkit toolkit = new Toolkit();
        toolkit.registerAgentTool(new StubTool("read_script", true));
        toolkit.registerAgentTool(new StubTool("update_script", false));

        PermissionContextState alwaysAsk = AgentToolPermissionPolicy.contextFor(
                toolkit, ToolExecutionMode.ALWAYS_ASK);
        PermissionContextState fullAccess = AgentToolPermissionPolicy.contextFor(
                toolkit, ToolExecutionMode.FULL_ACCESS);

        assertThat(alwaysAsk.getAskRules().keySet())
                .containsExactlyInAnyOrder("read_script", "update_script");
        assertThat(fullAccess.getMode()).isEqualTo(PermissionMode.BYPASS);
        assertThat(fullAccess.getAskRules()).isEmpty();
    }

    @Test
    void alwaysAllowStillAsksForDynamicDeleteRisk() {
        Toolkit toolkit = new Toolkit();
        AgentScopeToolAdapter tool = highRiskAdapter();
        toolkit.registerAgentTool(tool);

        PermissionContextState context = AgentToolPermissionPolicy.contextFor(
                toolkit, ToolExecutionMode.ALWAYS_ALLOW);

        assertThat(context.getMode()).isEqualTo(PermissionMode.BYPASS);
        assertThat(context.getAskRules()).containsKey("manage_script_scenes");
        assertThat(tool.matchRule(
                AgentToolPermissionPolicy.HIGH_RISK_RULE_CONTENT,
                Map.of("action", "delete"))).isTrue();
        assertThat(tool.matchRule(
                AgentToolPermissionPolicy.HIGH_RISK_RULE_CONTENT,
                Map.of("action", "add"))).isFalse();
    }

    @Test
    void alwaysAllowAsksForHighRiskNamedNonPlatformTools() {
        Toolkit toolkit = new Toolkit();
        toolkit.registerAgentTool(new StubTool("delete_remote_asset", false));

        PermissionContextState context = AgentToolPermissionPolicy.contextFor(
                toolkit, ToolExecutionMode.ALWAYS_ALLOW);

        assertThat(context.getAskRules()).containsKey("delete_remote_asset");
    }

    private AgentScopeToolAdapter highRiskAdapter() {
        ToolExecutor executor = new ToolExecutor() {
            @Override
            public String getToolName() {
                return "manage_script_scenes";
            }

            @Override
            public String getDisplayName() {
                return "管理场次";
            }

            @Override
            public String getToolDescription() {
                return "Add or delete script scenes";
            }

            @Override
            public String getParametersSchema() {
                return "{\"type\":\"object\",\"properties\":{}}";
            }

            @Override
            public ToolPermissionRisk getPermissionRisk(Map<String, Object> toolInput) {
                return "delete".equals(toolInput.get("action"))
                        ? ToolPermissionRisk.HIGH_RISK
                        : ToolPermissionRisk.EDIT;
            }

            @Override
            public boolean mayRequireHighRiskApproval() {
                return true;
            }

            @Override
            public String execute(
                    String toolInput,
                    ToolExecutionContext context) {
                return "{}";
            }
        };
        AgentScopeToolSchema.PreparedSchema schema = AgentScopeToolSchema.prepare(
                new ObjectMapper(), executor.getParametersSchema(), executor.getToolName());
        return new AgentScopeToolAdapter(
                executor,
                schema,
                Schedulers.immediate(),
                mock(RunLeaseGuard.class),
                new ObjectMapper());
    }

    private static final class StubTool extends ToolBase {
        private StubTool(String name, boolean readOnly) {
            super(ToolBase.builder()
                    .name(name)
                    .description(name)
                    .inputSchema(Map.of("type", "object", "properties", Map.of()))
                    .readOnly(readOnly));
        }

        @Override
        public Mono<ToolResultBlock> callAsync(ToolCallParam param) {
            return Mono.error(new UnsupportedOperationException("not executed"));
        }
    }
}
