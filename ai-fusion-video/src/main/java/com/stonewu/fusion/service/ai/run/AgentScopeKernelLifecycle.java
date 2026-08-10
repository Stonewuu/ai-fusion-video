package com.stonewu.fusion.service.ai.run;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.SmartLifecycle;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

@Component
@Slf4j
public final class AgentScopeKernelLifecycle implements SmartLifecycle {
    private final ObjectProvider<AgentRuntimeShutdownPort> shutdownPorts;
    private final Duration drainTimeout;
    private final AtomicBoolean running = new AtomicBoolean();
    private final AtomicReference<Mono<Void>> shutdownSignal = new AtomicReference<>();

    public AgentScopeKernelLifecycle(
            ObjectProvider<AgentRuntimeShutdownPort> shutdownPorts,
            @Value("${fusion.agentscope.shutdown-timeout:3s}") Duration drainTimeout) {
        this.shutdownPorts = Objects.requireNonNull(shutdownPorts, "shutdownPorts must not be null");
        this.drainTimeout = Objects.requireNonNull(drainTimeout, "drainTimeout must not be null");
        if (drainTimeout.isZero() || drainTimeout.isNegative()) {
            throw new IllegalArgumentException("drainTimeout must be greater than zero");
        }
    }

    @Override
    public void start() {
        running.set(true);
    }

    @Override
    public void stop() {
        stop(() -> { });
    }

    @Override
    public void stop(Runnable callback) {
        Objects.requireNonNull(callback, "callback must not be null");
        running.set(false);
        Mono<Void> shutdown = shutdownSignal.get();
        if (shutdown == null) {
            Mono<Void> candidate = Mono.defer(this::shutdownOperation)
                    .doOnError(failure -> log.error("AgentScope kernel shutdown failed", failure))
                    .cache();
            shutdown = shutdownSignal.compareAndSet(null, candidate)
                    ? candidate
                    : shutdownSignal.get();
        }
        shutdown.doFinally(ignored -> invokeCallback(callback))
                .subscribe(ignored -> { }, ignored -> { });
    }

    private Mono<Void> shutdownOperation() {
        AgentRuntimeShutdownPort shutdownPort = Objects.requireNonNull(
                shutdownPorts.getIfAvailable(),
                "AgentRuntimeShutdownPort must be available");
        return Objects.requireNonNull(shutdownPort.shutdown(drainTimeout),
                "AgentScope shutdown publisher must not be null");
    }

    private void invokeCallback(Runnable callback) {
        try {
            callback.run();
        } catch (Throwable failure) {
            log.error("AgentScope kernel shutdown callback failed", failure);
        }
    }

    @Override
    public boolean isRunning() {
        return running.get();
    }

    @Override
    public boolean isAutoStartup() {
        return true;
    }

    @Override
    public int getPhase() {
        return Integer.MAX_VALUE - 100;
    }
}
