package com.stonewu.fusion.service.ai.agentscope.kernel;

import io.agentscope.harness.agent.HarnessAgent;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

public final class AgentKernelResource implements AutoCloseable {
    private final HarnessAgent agent;
    private final OwnedChatModel ownedModel;
    private final AgentKernelToolkitResources toolResources;
    private final AtomicBoolean closed = new AtomicBoolean();

    public AgentKernelResource(
            HarnessAgent agent,
            OwnedChatModel ownedModel,
            AgentKernelToolkitResources toolResources) {
        this.agent = Objects.requireNonNull(agent, "agent must not be null");
        this.ownedModel = Objects.requireNonNull(ownedModel, "ownedModel must not be null");
        this.toolResources = Objects.requireNonNull(toolResources, "toolResources must not be null");
    }

    public HarnessAgent agent() {
        return agent;
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        Throwable failure = closeAndAccumulate(null, agent::close);
        failure = closeAndAccumulate(failure, toolResources::close);
        failure = closeAndAccumulate(failure, ownedModel::close);
        rethrow(failure);
    }

    static Throwable closeAndAccumulate(Throwable first, Runnable closeAction) {
        try {
            closeAction.run();
        } catch (Throwable failure) {
            if (first == null) {
                return failure;
            }
            if (failure != first) {
                first.addSuppressed(failure);
            }
        }
        return first;
    }

    static void rethrow(Throwable failure) {
        if (failure == null) {
            return;
        }
        if (failure instanceof RuntimeException runtimeException) {
            throw runtimeException;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        throw new IllegalStateException("Failed to close AgentScope kernel resource", failure);
    }
}
