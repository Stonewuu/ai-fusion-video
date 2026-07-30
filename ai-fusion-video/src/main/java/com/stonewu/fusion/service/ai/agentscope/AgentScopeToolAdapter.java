package com.stonewu.fusion.service.ai.agentscope;

import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.ToolPermissionRisk;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.CancellationContext;
import com.stonewu.fusion.service.ai.agentscope.tool.AbstractPlatformAgentTool;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.agentscope.permission.AgentToolPermissionPolicy;
import com.stonewu.fusion.service.ai.run.RunLeaseGuard;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.tool.ToolBase;
import io.agentscope.core.tool.ToolCallParam;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.Objects;

/** Adapts a platform tool to the strict AgentScope V2 {@link ToolBase} contract. */
public final class AgentScopeToolAdapter extends AbstractPlatformAgentTool {

    private final ToolExecutor toolExecutor;
    private final Scheduler toolScheduler;
    private final RunLeaseGuard leaseGuard;
    private final ObjectMapper objectMapper;

    public AgentScopeToolAdapter(
            ToolExecutor toolExecutor,
            AgentScopeToolSchema.PreparedSchema schema,
            Scheduler toolScheduler,
            RunLeaseGuard leaseGuard,
            ObjectMapper objectMapper) {
        super(builder(toolExecutor, schema));
        this.toolExecutor = Objects.requireNonNull(toolExecutor, "toolExecutor must not be null");
        this.toolScheduler = Objects.requireNonNull(toolScheduler, "toolScheduler must not be null");
        this.leaseGuard = Objects.requireNonNull(leaseGuard, "leaseGuard must not be null");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
    }

    @Override
    public Mono<ToolResultBlock> callAsync(ToolCallParam param) {
        return Mono.defer(() -> {
            RuntimeContext runtime = requireRuntimeContext(param);
            AgentRunContext run = requireContext(runtime, AgentRunContext.class);
            CancellationContext cancellation = requireContext(runtime, CancellationContext.class);
            com.stonewu.fusion.service.ai.agentscope.context.ToolExecutionContext toolContext =
                    requireContext(
                            runtime,
                            com.stonewu.fusion.service.ai.agentscope.context.ToolExecutionContext.class);
            Duration remaining = remaining(run);
            Map<String, Object> input = Objects.requireNonNull(
                    param.getInput(), "AgentScope tool input must not be null");
            Mono<String> invocation = Mono.fromCallable(() -> toolExecutor.execute(
                            JSONUtil.toJsonStr(input),
                            com.stonewu.fusion.service.ai.ToolExecutionContext.builder()
                                    .userId(toolContext.userId())
                                    .ownerType(toolContext.ownerType())
                                    .ownerId(toolContext.ownerId())
                                    .build()))
                    .subscribeOn(toolScheduler);
            return cancellation.checkpoint()
                    .then(assertLease(run))
                    .then(invocation)
                    .flatMap(result -> cancellation.checkpoint()
                            .then(assertLease(run))
                            .thenReturn(projectResult(param, result)))
                    .timeout(remaining);
        });
    }

    @Override
    public boolean matchRule(String ruleContent, Map<String, Object> toolInput) {
        if (AgentToolPermissionPolicy.HIGH_RISK_RULE_CONTENT.equals(ruleContent)) {
            return toolExecutor.getPermissionRisk(Objects.requireNonNull(
                    toolInput, "toolInput must not be null")) == ToolPermissionRisk.HIGH_RISK;
        }
        return super.matchRule(ruleContent, toolInput);
    }

    public boolean mayRequireHighRiskApproval() {
        return toolExecutor.mayRequireHighRiskApproval();
    }

    private ToolResultBlock projectResult(ToolCallParam param, String result) {
        if (result == null) {
            throw new IllegalStateException("AgentScope tool returned null output: " + getName());
        }
        try {
            JsonNode payload = objectMapper.readTree(result);
            if (payload != null && payload.isObject()) {
                JsonNode status = payload.get("status");
                if (status != null && status.isTextual()
                        && ("error".equalsIgnoreCase(status.textValue())
                            || "failed".equalsIgnoreCase(status.textValue()))) {
                    return errorResult(param, result);
                }
            }
        } catch (JsonProcessingException plainTextResult) {
            // ToolExecutor permits text output; only JSON objects carry the platform status contract.
        }
        return textResult(param, result);
    }

    private Mono<Void> assertLease(AgentRunContext run) {
        return leaseGuard.assertLease(run.runId(), run.ownerInstanceId(), run.ownerEpoch());
    }

    private RuntimeContext requireRuntimeContext(ToolCallParam param) {
        requireToolUse(param);
        if (param.getRuntimeContext() == null) {
            throw new IllegalArgumentException(
                    "AgentScope RuntimeContext is required for tool: " + getName());
        }
        return param.getRuntimeContext();
    }

    private <T> T requireContext(RuntimeContext runtime, Class<T> type) {
        T value = runtime.get(type);
        if (value == null) {
            throw new IllegalStateException(
                    "AgentScope RuntimeContext is missing " + type.getSimpleName()
                            + " for tool " + getName());
        }
        return value;
    }

    private Duration remaining(AgentRunContext run) {
        Duration remaining = Duration.between(Instant.now(), run.deadline());
        if (remaining.isZero() || remaining.isNegative()) {
            throw new IllegalStateException("Agent run deadline expired before tool execution");
        }
        return remaining;
    }

    private static ToolBase.Builder builder(
            ToolExecutor tool,
            AgentScopeToolSchema.PreparedSchema schema) {
        Objects.requireNonNull(tool, "toolExecutor must not be null");
        Objects.requireNonNull(schema, "schema must not be null");
        String description = tool.getToolDescription();
        if (description == null || description.isBlank()) {
            throw new IllegalArgumentException(
                    "AgentScope tool description must not be blank: " + tool.getToolName());
        }
        return ToolBase.builder()
                .name(tool.getToolName())
                .description(description)
                .inputSchema(schema.value())
                .readOnly(tool.isReadOnly())
                .concurrencySafe(tool.isConcurrencySafe());
    }
}
