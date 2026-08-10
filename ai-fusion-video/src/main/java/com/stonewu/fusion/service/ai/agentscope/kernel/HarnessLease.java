package com.stonewu.fusion.service.ai.agentscope.kernel;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

public final class HarnessLease implements AutoCloseable {
    private final AgentKernelResource resource;
    private final Runnable release;
    private final AtomicBoolean closed = new AtomicBoolean();

    HarnessLease(AgentKernelResource resource, Runnable release) {
        this.resource = Objects.requireNonNull(resource, "resource must not be null");
        this.release = Objects.requireNonNull(release, "release must not be null");
    }

    public AgentKernelResource resource() {
        return resource;
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            release.run();
        }
    }
}
