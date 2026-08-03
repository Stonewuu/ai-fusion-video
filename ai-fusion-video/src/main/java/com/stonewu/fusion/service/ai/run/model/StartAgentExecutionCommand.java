package com.stonewu.fusion.service.ai.run.model;

import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextRequest;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import io.agentscope.core.message.Msg;

import java.util.List;
import java.util.Objects;

public record StartAgentExecutionCommand(
        StartedAgentRun run,
        List<Msg> messages,
        AgentKernelSnapshot kernelSnapshot,
        AgentKernelSpec kernelSpec,
        AgentScopeRuntimeContextRequest runtimeContextRequest) {

    public StartAgentExecutionCommand {
        run = Objects.requireNonNull(run, "run must not be null");
        messages = List.copyOf(Objects.requireNonNull(messages, "messages must not be null"));
        if (messages.isEmpty()) {
            throw new IllegalArgumentException("messages must not be empty");
        }
        kernelSnapshot = Objects.requireNonNull(kernelSnapshot, "kernelSnapshot must not be null");
        kernelSpec = Objects.requireNonNull(kernelSpec, "kernelSpec must not be null");
        runtimeContextRequest = Objects.requireNonNull(
                runtimeContextRequest, "runtimeContextRequest must not be null");
    }
}
