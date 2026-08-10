package com.stonewu.fusion.service.ai.agentscope;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.CancellationContext;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.run.RunLeaseGuard;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolResultState;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.tool.ToolCallParam;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.time.Instant;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AgentScopeToolAdapterTests {

    @Test
    void callAsyncShouldPreserveToolCallIdAndName() {
        ToolExecutor toolExecutor = mock(ToolExecutor.class);
        when(toolExecutor.getToolName()).thenReturn("asset_query");
        when(toolExecutor.getToolDescription()).thenReturn("query asset");
        when(toolExecutor.execute(eq("{\"keyword\":\"cat\"}"), any(ToolExecutionContext.class)))
                .thenReturn("ok");
        RunLeaseGuard leaseGuard = mock(RunLeaseGuard.class);
        when(leaseGuard.assertLease("run-1", "owner-1", 1L)).thenReturn(Mono.empty());

        AgentScopeToolAdapter adapter = new AgentScopeToolAdapter(
                toolExecutor,
                AgentScopeToolSchema.prepare(
                        new ObjectMapper(),
                        "{\"type\":\"object\",\"properties\":{\"keyword\":{\"type\":\"string\"}}}",
                        "asset_query"),
                Schedulers.immediate(),
                leaseGuard,
                new ObjectMapper());

        ToolUseBlock toolUseBlock = ToolUseBlock.builder()
                .id("call-123")
                .name("asset_query")
                .input(Map.of("keyword", "cat"))
                .build();

        RuntimeContext runtime = RuntimeContext.builder()
                .put(AgentRunContext.class, new AgentRunContext(
                        "run-1", "owner-1", 1L, Instant.now().plusSeconds(30)))
                .put(CancellationContext.class, CancellationContext.noop())
                .put(com.stonewu.fusion.service.ai.agentscope.context.ToolExecutionContext.class,
                        new com.stonewu.fusion.service.ai.agentscope.context.ToolExecutionContext(
                                42L, 1, 42L))
                .build();
        ToolCallParam param = ToolCallParam.builder()
                .toolUseBlock(toolUseBlock)
                .input(Map.of("keyword", "cat"))
                .runtimeContext(runtime)
                .build();

        ToolResultBlock result = adapter.callAsync(param).block();

        assertEquals("call-123", result.getId());
        assertEquals("asset_query", result.getName());
                TextBlock output = assertInstanceOf(TextBlock.class, result.getOutput().getFirst());
                assertEquals("ok", output.getText());
    }

    @Test
    void callAsyncShouldExposePlatformErrorJsonAsAgentScopeError() {
        ToolExecutor toolExecutor = mock(ToolExecutor.class);
        when(toolExecutor.getToolName()).thenReturn("asset_query");
        when(toolExecutor.getToolDescription()).thenReturn("query asset");
        when(toolExecutor.execute(eq("{}"), any(ToolExecutionContext.class)))
                .thenReturn("{\"status\":\"error\",\"message\":\"asset missing\"}");
        RunLeaseGuard leaseGuard = mock(RunLeaseGuard.class);
        when(leaseGuard.assertLease("run-1", "owner-1", 1L)).thenReturn(Mono.empty());
        AgentScopeToolAdapter adapter = new AgentScopeToolAdapter(
                toolExecutor,
                AgentScopeToolSchema.prepare(
                        new ObjectMapper(),
                        "{\"type\":\"object\",\"properties\":{}}",
                        "asset_query"),
                Schedulers.immediate(),
                leaseGuard,
                new ObjectMapper());
        ToolUseBlock toolUseBlock = ToolUseBlock.builder()
                .id("call-error")
                .name("asset_query")
                .input(Map.of())
                .build();
        RuntimeContext runtime = RuntimeContext.builder()
                .put(AgentRunContext.class, new AgentRunContext(
                        "run-1", "owner-1", 1L, Instant.now().plusSeconds(30)))
                .put(CancellationContext.class, CancellationContext.noop())
                .put(com.stonewu.fusion.service.ai.agentscope.context.ToolExecutionContext.class,
                        new com.stonewu.fusion.service.ai.agentscope.context.ToolExecutionContext(
                                42L, 1, 42L))
                .build();

        ToolResultBlock result = adapter.callAsync(ToolCallParam.builder()
                        .toolUseBlock(toolUseBlock)
                        .input(Map.of())
                        .runtimeContext(runtime)
                        .build())
                .block();

        assertEquals(ToolResultState.ERROR, result.getState());
        TextBlock output = assertInstanceOf(TextBlock.class, result.getOutput().getFirst());
        assertEquals(
                "Error: {\"status\":\"error\",\"message\":\"asset missing\"}",
                output.getText());
    }
}
