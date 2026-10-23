package com.stonewu.fusion.service.ai.agentscope.mcp;

import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.AuthenticatedUserContext;
import com.stonewu.fusion.service.ai.agentscope.context.CancellationContext;
import com.stonewu.fusion.service.ai.run.RunLeaseGuard;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.tool.ToolBase;
import io.agentscope.core.tool.ToolCallParam;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import reactor.test.StepVerifier;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class GuardedMcpAgentToolTests {
    private final AtomicInteger calls = new AtomicInteger();
    private final RunLeaseGuard leases = mock(RunLeaseGuard.class);
    private final ToolResultBlock result = ToolResultBlock.of("call", "external_search", List.of(TextBlock.builder().text("实际 MCP 结果").build()));

    @Test void preservesMcpMetadataAndReturnsTheOriginalTypedResultOnce() {
        when(leases.assertLease("run", "owner", 7L)).thenReturn(Mono.empty());
        var tool = guarded(Mono.just(result), () -> {});
        assertThat(tool.isMcp()).isTrue();
        assertThat(tool.getMcpName()).isEqualTo("local-test");
        assertThat(tool.isReadOnly()).isTrue();
        assertThat(tool.isConcurrencySafe()).isTrue();
        assertThat(tool.getStrict()).isTrue();
        assertThat(tool.getOutputSchema()).containsEntry("type", "object");
        StepVerifier.create(tool.callAsync(param(42L, CancellationContext.noop(), Instant.now().plusSeconds(30))))
                .expectNext(result).verifyComplete();
        assertThat(calls).hasValue(1);
    }

    @Test void rejectsAnotherUsersToolBeforeCallingTheRemoteService() {
        StepVerifier.create(guarded(Mono.just(result), () -> {}).callAsync(param(43L, CancellationContext.noop(), Instant.now().plusSeconds(30))))
                .expectErrorMessage("MCP 工具不属于当前用户").verify();
        assertThat(calls).hasValue(0);
    }

    @Test void staleLeaseAndChangedConfigurationCannotSendAnMcpRequest() {
        when(leases.assertLease("run", "owner", 7L)).thenReturn(Mono.error(new IllegalStateException("失去租约")));
        StepVerifier.create(guarded(Mono.just(result), () -> {}).callAsync(param(42L, CancellationContext.noop(), Instant.now().plusSeconds(30))))
                .expectErrorMessage("失去租约").verify();
        when(leases.assertLease("run", "owner", 7L)).thenReturn(Mono.empty());
        StepVerifier.create(guarded(Mono.just(result), () -> { throw new IllegalStateException("配置已变化"); })
                        .callAsync(param(42L, CancellationContext.noop(), Instant.now().plusSeconds(30))))
                .expectErrorMessage("配置已变化").verify();
        assertThat(calls).hasValue(0);
    }

    @Test void cancelledOrExpiredRunCannotSendAnMcpRequest() {
        var cancellation = new CancellationContext(() -> Mono.error(new IllegalStateException("已取消")));
        StepVerifier.create(guarded(Mono.just(result), () -> {}).callAsync(param(42L, cancellation, Instant.now().plusSeconds(30))))
                .expectErrorMessage("已取消").verify();
        StepVerifier.create(guarded(Mono.just(result), () -> {}).callAsync(param(42L, CancellationContext.noop(), Instant.now().minusSeconds(1))))
                .expectErrorMessage("智能体运行期限已过").verify();
        assertThat(calls).hasValue(0);
    }

    @Test void cancellationDuringRemoteCallDoesNotPublishItsLateResultOrRepeatTheRequest() {
        when(leases.assertLease("run", "owner", 7L)).thenReturn(Mono.empty());
        var cancelled = new AtomicBoolean();
        var cancellation = new CancellationContext(() -> cancelled.get() ? Mono.error(new IllegalStateException("已取消")) : Mono.empty());
        var response = Mono.fromSupplier(() -> { cancelled.set(true); return result; });
        StepVerifier.create(guarded(response, () -> {}).callAsync(param(42L, cancellation, Instant.now().plusSeconds(30))))
                .expectErrorMessage("已取消").verify();
        assertThat(calls).hasValue(1);
    }

    private GuardedMcpAgentTool guarded(Mono<ToolResultBlock> response, Runnable configurationCheck) {
        var delegate = new ToolBase(ToolBase.builder().name("external_search").description("已授权的外部素材查询")
                .inputSchema(Map.of("type", "object", "properties", Map.of())).readOnly(true).concurrencySafe(true).mcp("local-test")) {
            @Override public Mono<ToolResultBlock> callAsync(ToolCallParam param) { calls.incrementAndGet(); return response; }
            @Override public Boolean getStrict() { return true; }
            @Override public Map<String, Object> getOutputSchema() { return Map.of("type", "object"); }
        };
        return new GuardedMcpAgentTool(delegate, 42L, leases, configurationCheck, Schedulers.immediate());
    }

    private ToolCallParam param(long user, CancellationContext cancellation, Instant deadline) {
        var context = RuntimeContext.builder().userId(Long.toString(user)).sessionId("session")
                .put(AuthenticatedUserContext.class, new AuthenticatedUserContext(user))
                .put(AgentRunContext.class, new AgentRunContext("run", "owner", 7L, deadline))
                .put(CancellationContext.class, cancellation).build();
        return ToolCallParam.builder().runtimeContext(context)
                .toolUseBlock(ToolUseBlock.builder().id("call").name("external_search").input(Map.of()).build()).build();
    }
}
