package com.stonewu.fusion.service.ai.agentscope.kernel;

@FunctionalInterface
public interface AgentKernelToolkitResources extends AutoCloseable {

    static AgentKernelToolkitResources none() {
        return () -> { };
    }

    @Override
    void close();
}
