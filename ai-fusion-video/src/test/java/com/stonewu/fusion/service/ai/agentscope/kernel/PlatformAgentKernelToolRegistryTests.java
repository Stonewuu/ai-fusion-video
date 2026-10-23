package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.AiAgentService;
import com.stonewu.fusion.config.ai.AiAgentRegistry;
import com.stonewu.fusion.service.ai.configuration.AgentConfigurationTestSupport;
import com.stonewu.fusion.service.ai.AiToolConfigService;
import com.stonewu.fusion.service.ai.ToolExecutorRegistry;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.AuthenticatedUserContext;
import com.stonewu.fusion.service.ai.agentscope.context.CancellationContext;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentScopeMcpRegistry;
import com.stonewu.fusion.service.ai.agentscope.mcp.GuardedMcpAgentTool;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRunPort;
import com.stonewu.fusion.service.ai.run.RunLeaseGuard;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.tool.ToolBase;
import io.agentscope.core.tool.ToolCallParam;
import io.agentscope.core.tool.Toolkit;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.ObjectProvider;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import reactor.test.StepVerifier;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PlatformAgentKernelToolRegistryTests {
    @Test void registeredMcpToolUsesTheGuardAndRechecksItsCapturedConfiguration() throws Exception {
        var json = new ObjectMapper();
        var mcp = mock(AgentScopeMcpRegistry.class);
        var leases = mock(RunLeaseGuard.class);
        var schedulers = mock(AgentRuntimeSchedulers.class);
        when(schedulers.toolBlocking()).thenReturn(Schedulers.immediate());
        when(leases.assertLease("run", "owner", 1L)).thenReturn(Mono.empty());
        var calls = new AtomicInteger();
        var delegate = new ToolBase(ToolBase.builder().name("external_search").description("查询素材")
                .inputSchema(Map.of("type", "object", "properties", Map.of())).readOnly(true).concurrencySafe(false).mcp("library")) {
            @Override public Mono<ToolResultBlock> callAsync(ToolCallParam param) {
                calls.incrementAndGet();
                return Mono.just(ToolResultBlock.of("call", getName(), List.of(TextBlock.builder().text("result").build())));
            }
        };
        var schema = AgentScopeToolSchema.prepare(json, json.writeValueAsString(delegate.getParameters()), delegate.getName());
        var spec = mock(AgentKernelSpec.class);
        when(spec.agentDefinitionStableKey()).thenReturn("canvas_assistant");
        when(spec.promptVariables()).thenReturn(Map.of(AgentKernelSpecFactory.OWNER_USER_ID_VARIABLE, "42",
                AgentKernelSpecFactory.MCP_CONFIG_FINGERPRINT_VARIABLE, "captured"));
        when(spec.toolWhitelist()).thenReturn(Set.of(delegate.getName()));
        when(spec.toolManifest()).thenReturn(List.of(new AgentKernelToolManifest(delegate.getName(),
                AgentKernelToolManifest.schemaSha256(schema.canonicalJson()), true, false)));
        when(mcp.isMcpTool(delegate.getName(), "canvas_assistant", 42L)).thenReturn(true);
        when(mcp.userConfigurationFingerprint(42L)).thenReturn("captured");
        when(mcp.register(eq("canvas_assistant"), eq(42L), eq(Set.of(delegate.getName())), any(Toolkit.class)))
                .thenAnswer(invocation -> {
                    Toolkit toolkit = invocation.getArgument(3);
                    toolkit.registerAgentTool(delegate);
                    return AgentKernelToolkitResources.none();
                });
        ObjectProvider<PlatformSubAgentRunPort> children = mock(ObjectProvider.class);
        var registry = new PlatformAgentKernelToolRegistry(mock(ToolExecutorRegistry.class), mock(AiToolConfigService.class),
                AgentConfigurationTestSupport.agents(new AiAgentRegistry()), mock(AgentKernelSpecFactory.class), children, schedulers, leases, json, mcp, List.of());
        var toolkit = new Toolkit();
        registry.register(spec, toolkit);
        assertThat(toolkit.getTool(delegate.getName())).isInstanceOf(GuardedMcpAgentTool.class);
        var context = RuntimeContext.builder().userId("42").sessionId("session")
                .put(AuthenticatedUserContext.class, new AuthenticatedUserContext(42L))
                .put(AgentRunContext.class, new AgentRunContext("run", "owner", 1L, Instant.now().plusSeconds(30)))
                .put(CancellationContext.class, CancellationContext.noop()).build();
        var param = ToolCallParam.builder().runtimeContext(context)
                .toolUseBlock(ToolUseBlock.builder().id("call").name(delegate.getName()).input(Map.of()).build()).build();
        StepVerifier.create(toolkit.getTool(delegate.getName()).callAsync(param)).expectNextCount(1).verifyComplete();
        when(mcp.userConfigurationFingerprint(42L)).thenReturn("changed");
        StepVerifier.create(toolkit.getTool(delegate.getName()).callAsync(param))
                .expectErrorMessage("MCP 配置已变化，请重新启动任务").verify();
        assertThat(calls).hasValue(1);
    }
}
