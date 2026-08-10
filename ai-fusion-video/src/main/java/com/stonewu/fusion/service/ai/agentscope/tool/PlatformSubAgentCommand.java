package com.stonewu.fusion.service.ai.agentscope.tool;

import com.stonewu.fusion.service.ai.agentscope.context.ProjectContext;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import io.agentscope.core.message.Msg;

import java.time.Instant;
import java.util.List;
import java.util.Objects;

public record PlatformSubAgentCommand(
        String parentRunId,
        String parentToolCallId,
        String parentOwnerInstanceId,
        long parentOwnerEpoch,
        String agentName,
        AgentKernelSpec kernelSpec,
        List<Msg> messages,
        ProjectContext projectContext,
        ToolExecutionMode toolExecutionMode,
        Instant deadline) {

    public PlatformSubAgentCommand {
        requireText(parentRunId, "parentRunId");
        requireText(parentToolCallId, "parentToolCallId");
        requireText(parentOwnerInstanceId, "parentOwnerInstanceId");
        if (parentOwnerEpoch <= 0) {
            throw new IllegalArgumentException("parentOwnerEpoch must be positive");
        }
        requireText(agentName, "agentName");
        kernelSpec = Objects.requireNonNull(kernelSpec, "kernelSpec must not be null");
        messages = List.copyOf(Objects.requireNonNull(messages, "messages must not be null"));
        if (messages.isEmpty()) {
            throw new IllegalArgumentException("messages must not be empty");
        }
        toolExecutionMode = Objects.requireNonNull(
                toolExecutionMode, "toolExecutionMode must not be null");
        deadline = Objects.requireNonNull(deadline, "deadline must not be null");
    }

    private static void requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
    }
}
