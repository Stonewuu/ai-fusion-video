package com.stonewu.fusion.service.ai.agentscope.mcp;

import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.AuthenticatedUserContext;
import com.stonewu.fusion.service.ai.agentscope.context.CancellationContext;
import com.stonewu.fusion.service.ai.run.AgentExecutionLimits;
import com.stonewu.fusion.service.ai.run.RunLeaseGuard;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.tool.AgentTool;
import io.agentscope.core.tool.ToolBase;
import io.agentscope.core.tool.ToolCallParam;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;

import java.util.Map;
import java.util.Objects;

/**
 * MCP invocations share the platform's cancellation, owner and lease boundary.
 */
public final class GuardedMcpAgentTool extends ToolBase {
    private final AgentTool delegate;
    private final Long ownerUserId;
    private final RunLeaseGuard leases;
    private final Runnable configurationCheck;
    private final Scheduler scheduler;

    public GuardedMcpAgentTool(AgentTool delegate, Long ownerUserId, RunLeaseGuard leases,
                               Runnable configurationCheck, Scheduler scheduler) {
        super(metadata(delegate));
        this.delegate = Objects.requireNonNull(delegate);
        this.ownerUserId = ownerUserId;
        this.leases = Objects.requireNonNull(leases);
        this.configurationCheck = Objects.requireNonNull(configurationCheck);
        this.scheduler = Objects.requireNonNull(scheduler);
    }

    @Override
    public Mono<ToolResultBlock> callAsync(ToolCallParam param) {
        return Mono.defer(() -> {
            var context = Objects.requireNonNull(param.getRuntimeContext(), "MCP 调用缺少运行上下文");
            var user = Objects.requireNonNull(context.get(AuthenticatedUserContext.class), "MCP 调用缺少认证用户");
            var run = Objects.requireNonNull(context.get(AgentRunContext.class), "MCP 调用缺少运行身份");
            var cancellation = Objects.requireNonNull(context.get(CancellationContext.class), "MCP 调用缺少取消检查");
            if (ownerUserId != null && ownerUserId.longValue() != user.userId()) {
                throw new IllegalStateException("MCP 工具不属于当前用户");
            }
            var call = Objects.requireNonNull(param.getToolUseBlock(), "MCP 调用缺少工具身份");
            if (!getName().equals(call.getName()) || call.getId() == null || call.getId().isBlank()) {
                throw new IllegalArgumentException("MCP 工具调用身份不一致");
            }
            return checkpoint(run, cancellation)
                    .then(Mono.fromRunnable(configurationCheck).subscribeOn(scheduler))
                    .then(checkpoint(run, cancellation))
                    .then(Mono.defer(() -> delegate.callAsync(param)))
                    .flatMap(result -> checkpoint(run, cancellation).thenReturn(result))
                    .transform(action -> AgentExecutionLimits.untilDeadline(action, run.deadline()));
        });
    }

    private Mono<Void> checkpoint(AgentRunContext run, CancellationContext cancellation) {
        return cancellation.checkpoint()
                .then(Mono.defer(() -> leases.assertLease(run.runId(), run.ownerInstanceId(), run.ownerEpoch())));
    }

    @Override
    public Boolean getStrict() {
        return delegate.getStrict();
    }

    @Override
    public Map<String, Object> getOutputSchema() {
        return delegate.getOutputSchema();
    }

    @Override
    public boolean matchRule(String content, Map<String, Object> input) {
        return delegate instanceof ToolBase base ? base.matchRule(content, input) : super.matchRule(content, input);
    }

    private static Builder metadata(AgentTool tool) {
        var builder = ToolBase.builder().name(tool.getName()).description(tool.getDescription())
                .inputSchema(tool.getParameters()).readOnly(tool.isReadOnly()).concurrencySafe(false);
        if (tool instanceof ToolBase base) {
            builder.concurrencySafe(base.isConcurrencySafe()).externalTool(base.isExternalTool())
                    .stateInjected(base.isStateInjected());
            if (base.isMcp()) {
                builder.mcp(base.getMcpName());
            }
        }
        return builder;
    }
}
