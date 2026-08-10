package com.stonewu.fusion.service.ai.agentscope.kernel;

@FunctionalInterface
public interface AgentKernelModelFactory {
    OwnedChatModel create(AgentKernelSpec spec);
}
