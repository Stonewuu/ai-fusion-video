package com.stonewu.fusion.service.ai.agentscope.permission;

import com.stonewu.fusion.service.ai.agentscope.AgentScopeToolAdapter;
import com.stonewu.fusion.service.ai.agentscope.context.ToolPermissionContext;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.permission.PermissionBehavior;
import io.agentscope.core.permission.PermissionContextState;
import io.agentscope.core.permission.PermissionMode;
import io.agentscope.core.permission.PermissionRule;
import io.agentscope.core.tool.AgentTool;
import io.agentscope.core.tool.Toolkit;
import io.agentscope.harness.agent.HarnessAgent;

import java.util.Locale;
import java.util.Objects;

/** Builds and applies the platform's four user-facing AgentScope permission policies. */
public final class AgentToolPermissionPolicy {

    public static final String HIGH_RISK_RULE_CONTENT = "afv:risk:high";
    private static final String RULE_SOURCE = "platformPolicy";

    private AgentToolPermissionPolicy() {
    }

    public static PermissionContextState contextFor(
            Toolkit toolkit, ToolExecutionMode executionMode) {
        Objects.requireNonNull(toolkit, "toolkit must not be null");
        ToolExecutionMode safeMode = Objects.requireNonNull(
                executionMode, "executionMode must not be null");
        PermissionContextState.Builder builder = PermissionContextState.builder()
                .mode(safeMode == ToolExecutionMode.ALWAYS_ALLOW
                        || safeMode == ToolExecutionMode.FULL_ACCESS
                        ? PermissionMode.BYPASS
                        : PermissionMode.DEFAULT);

        for (String toolName : toolkit.getToolNames()) {
            AgentTool tool = Objects.requireNonNull(
                    toolkit.getTool(toolName), "Toolkit lost registered tool: " + toolName);
            switch (safeMode) {
                case DEFAULT -> {
                    if (tool.isReadOnly()) {
                        builder.addAllowRule(toolName, rule(
                                toolName, null, PermissionBehavior.ALLOW));
                    } else {
                        builder.addAskRule(toolName, rule(
                                toolName, null, PermissionBehavior.ASK));
                    }
                }
                case ALWAYS_ASK -> builder.addAskRule(toolName, rule(
                        toolName, null, PermissionBehavior.ASK));
                case ALWAYS_ALLOW -> {
                    if (tool instanceof AgentScopeToolAdapter platformTool
                            && platformTool.mayRequireHighRiskApproval()) {
                        builder.addAskRule(toolName, rule(
                                toolName, HIGH_RISK_RULE_CONTENT, PermissionBehavior.ASK));
                    } else if (hasHighRiskName(toolName)) {
                        builder.addAskRule(toolName, rule(
                                toolName, null, PermissionBehavior.ASK));
                    }
                }
                case FULL_ACCESS -> {
                    // BYPASS with no platform ask rules intentionally allows every tool.
                }
            }
        }
        return builder.build();
    }

    public static void applyRequestedPolicy(
            HarnessAgent agent, RuntimeContext runtimeContext) {
        Objects.requireNonNull(agent, "agent must not be null");
        Objects.requireNonNull(runtimeContext, "runtimeContext must not be null");
        ToolPermissionContext requested = Objects.requireNonNull(
                runtimeContext.get(ToolPermissionContext.class),
                "RuntimeContext is missing ToolPermissionContext");
        PermissionContextState context = contextFor(agent.getToolkit(), requested.mode());
        agent.getDelegate()
                .getAgentState(runtimeContext.getUserId(), runtimeContext.getSessionId())
                .setPermissionContext(context);
        // Rebuilds AgentScope's per-session PermissionEngine and persists the new context.
        agent.setPermissionMode(runtimeContext, context.getMode());
    }

    private static PermissionRule rule(
            String toolName, String content, PermissionBehavior behavior) {
        return new PermissionRule(toolName, content, behavior, RULE_SOURCE);
    }

    private static boolean hasHighRiskName(String toolName) {
        String normalized = toolName.toLowerCase(Locale.ROOT);
        return normalized.contains("delete")
                || normalized.contains("remove")
                || normalized.contains("destroy")
                || normalized.contains("drop")
                || normalized.contains("purge");
    }
}
