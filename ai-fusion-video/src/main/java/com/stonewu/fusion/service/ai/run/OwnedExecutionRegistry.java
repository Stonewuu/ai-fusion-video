package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.ExecutionStopReason;
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Autowired;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;

@Component
public final class OwnedExecutionRegistry {

    private final ConcurrentHashMap<String, AgentExecutionHandle> handles =
            new ConcurrentHashMap<>();
    private final AtomicReference<Sinks.Empty<Void>> emptyWaiter = new AtomicReference<>();
    private final ObjectMapper objectMapper;
    private final int maxEvents;
    private final long maxBytes;
    private final Scheduler deadlineScheduler;
    private final Clock clock;

    @Autowired
    public OwnedExecutionRegistry(ObjectMapper objectMapper, AgentScopeV2Properties properties) {
        this(
                objectMapper,
                properties.getIngress().getMaxEvents(),
                properties.getIngress().getMaxBytes(),
                Schedulers.parallel(),
                Clock.systemUTC());
    }

    OwnedExecutionRegistry(
            ObjectMapper objectMapper,
            int maxEvents,
            long maxBytes,
            Scheduler deadlineScheduler,
            Clock clock) {
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
        this.maxEvents = maxEvents;
        this.maxBytes = maxBytes;
        this.deadlineScheduler = Objects.requireNonNull(
                deadlineScheduler, "deadlineScheduler must not be null");
        this.clock = Objects.requireNonNull(clock, "clock must not be null");
    }

    public Mono<Void> registerAndLaunch(
            AgentExecution execution,
            Instant deadline,
            Function<Flux<AgentEventEnvelope>, Flux<AgentEventEnvelope>> ingressTransform,
            Function<Flux<AgentEventEnvelope>, Mono<Void>> durableConsumer,
            Function<AgentExecutionHandle.Outcome, Mono<Void>> completionAction) {
        return registerAndLaunch(
                execution,
                deadline,
                ingressTransform,
                durableConsumer,
                ignored -> Mono.never(),
                completionAction);
    }

    public Mono<Void> registerAndLaunch(
            AgentExecution execution,
            Instant deadline,
            Function<Flux<AgentEventEnvelope>, Flux<AgentEventEnvelope>> ingressTransform,
            Function<Flux<AgentEventEnvelope>, Mono<Void>> durableConsumer,
            Function<AgentExecutionHandle, Mono<Void>> controlMonitor,
            Function<AgentExecutionHandle.Outcome, Mono<Void>> completionAction) {
        return Mono.fromRunnable(() -> {
            Objects.requireNonNull(execution, "execution must not be null");
            AtomicReference<AgentExecutionHandle> reference = new AtomicReference<>();
            AgentExecutionHandle handle = new AgentExecutionHandle(
                    execution,
                    new BoundedAgentEventIngress(objectMapper, maxEvents, maxBytes),
                    deadline,
                    deadlineScheduler,
                    clock,
                    () -> remove(execution.runId(), reference.get()));
            reference.set(handle);
            AgentExecutionHandle existing = handles.putIfAbsent(execution.runId(), handle);
            if (existing != null) {
                handle.close();
                throw new ExecutionAlreadyOwnedException(execution.runId());
            }
            try {
                Mono<Void> monitor = Objects.requireNonNull(
                        controlMonitor.apply(handle),
                        "controlMonitor returned null");
                handle.launch(
                        ingressTransform,
                        durableConsumer,
                        monitor,
                        completionAction);
            } catch (Throwable launchFailure) {
                handle.close();
                throw launchFailure;
            }
        });
    }

    public Mono<Boolean> interruptOwned(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            ExecutionStopReason reason) {
        return Mono.defer(() -> {
            AgentExecutionHandle handle = handles.get(runId);
            if (handle == null
                    || !handle.ownerInstanceId().equals(ownerInstanceId)
                    || handle.ownerEpoch() != ownerEpoch) {
                return Mono.just(false);
            }
            return handle.interrupt(reason);
        });
    }

    public List<AgentExecutionHandle> snapshot() {
        return List.copyOf(handles.values());
    }

    public int size() {
        return handles.size();
    }

    public Mono<Void> awaitEmpty(Duration timeout) {
        Objects.requireNonNull(timeout, "timeout must not be null");
        if (timeout.isNegative() || timeout.isZero()) {
            return Mono.error(new IllegalArgumentException("timeout must be greater than zero"));
        }
        return Mono.defer(() -> {
            if (handles.isEmpty()) {
                return Mono.empty();
            }
            Sinks.Empty<Void> candidate = Sinks.empty();
            emptyWaiter.compareAndSet(null, candidate);
            Sinks.Empty<Void> waiter = emptyWaiter.get();
            if (handles.isEmpty()) {
                signalEmpty();
                return Mono.empty();
            }
            return waiter.asMono().timeout(timeout, Mono.error(new TimeoutException(
                    "Timed out waiting for owned Agent executions to drain")));
        });
    }

    private void remove(String runId, AgentExecutionHandle expected) {
        if (expected != null && handles.remove(runId, expected) && handles.isEmpty()) {
            signalEmpty();
        }
    }

    private void signalEmpty() {
        Sinks.Empty<Void> waiter = emptyWaiter.getAndSet(null);
        if (waiter != null) {
            waiter.tryEmitEmpty();
        }
    }

    public static final class ExecutionAlreadyOwnedException extends RuntimeException {
        private ExecutionAlreadyOwnedException(String runId) {
            super("Agent run already has an owned execution: " + runId);
        }
    }
}
