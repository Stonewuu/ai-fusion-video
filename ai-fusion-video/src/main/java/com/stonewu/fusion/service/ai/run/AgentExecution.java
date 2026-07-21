package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.ExecutionStopReason;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;

/** A cold execution stream plus explicit interruption and one-shot resource cleanup. */
public final class AgentExecution implements AutoCloseable {

    private final String runId;
    private final String ownerInstanceId;
    private final long ownerEpoch;
    private final long userId;
    private final String stateSessionId;
    private final Flux<AgentEventEnvelope> events;
    private final Function<ExecutionStopReason, Mono<Void>> interruptAction;
    private final Runnable closeAction;
    private final AtomicBoolean closed = new AtomicBoolean();

    public AgentExecution(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            long userId,
            String stateSessionId,
            Flux<AgentEventEnvelope> events,
            Function<ExecutionStopReason, Mono<Void>> interruptAction,
            Runnable closeAction) {
        this.runId = requireText(runId, "runId");
        this.ownerInstanceId = requireText(ownerInstanceId, "ownerInstanceId");
        if (ownerEpoch <= 0) {
            throw new IllegalArgumentException("ownerEpoch must be positive");
        }
        if (userId <= 0) {
            throw new IllegalArgumentException("userId must be positive");
        }
        this.ownerEpoch = ownerEpoch;
        this.userId = userId;
        this.stateSessionId = requireText(stateSessionId, "stateSessionId");
        this.events = Objects.requireNonNull(events, "events must not be null");
        this.interruptAction = Objects.requireNonNull(
                interruptAction, "interruptAction must not be null");
        this.closeAction = Objects.requireNonNull(closeAction, "closeAction must not be null");
    }

    public String runId() {
        return runId;
    }

    public String ownerInstanceId() {
        return ownerInstanceId;
    }

    public long ownerEpoch() {
        return ownerEpoch;
    }

    public long userId() {
        return userId;
    }

    public String stateSessionId() {
        return stateSessionId;
    }

    public Flux<AgentEventEnvelope> events() {
        return events;
    }

    public Mono<Void> interrupt(ExecutionStopReason reason) {
        Objects.requireNonNull(reason, "reason must not be null");
        return Mono.defer(() -> Objects.requireNonNull(
                interruptAction.apply(reason), "interruptAction returned null"));
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            closeAction.run();
        }
    }

    public boolean isClosed() {
        return closed.get();
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }
}
