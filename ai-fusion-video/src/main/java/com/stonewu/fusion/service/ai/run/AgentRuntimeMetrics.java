package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.enums.ai.AgentRunStatus;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.springframework.beans.factory.ObjectProvider;

import java.time.Duration;
import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

/** Low-cardinality Micrometer facade owned by the durable Agent runtime. */
public final class AgentRuntimeMetrics {

    public static final String PREFIX = "fusion.agentscope.runtime";
    private static final Set<AgentRunStatus> TERMINAL_STATUSES = Set.of(
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED);
    private static final AgentRuntimeMetrics NOOP =
            new AgentRuntimeMetrics(new SimpleMeterRegistry());

    private final AtomicLong activeRuns = new AtomicLong();
    private final AtomicLong waitingRuns = new AtomicLong();
    private final AtomicLong activeHarnessLeases = new AtomicLong();
    private final AtomicLong outboxBacklog = new AtomicLong();
    private final Counter ownerLost;
    private final Map<AgentRunStatus, Counter> terminals;
    private final Counter harnessHit;
    private final Counter harnessMiss;
    private final Counter harnessEviction;
    private final Counter harnessCapacityRejected;
    private final Timer stateLatency;
    private final Counter stateFailure;
    private final Counter stateBulkheadRejected;
    private final Timer eventPersistLatency;
    private final Counter eventPersisted;
    private final Counter sequenceAllocated;
    private final Counter eventBackpressureRejected;
    private final Counter outboxRetry;
    private final Map<AgentRunStatus, Counter> providerTerminals;
    private final Counter providerClosed;
    private final Counter toolSchedulerViolation;
    private final Counter replayDelivered;
    private final Counter replayDuplicateSuppressed;
    private final Counter replayTerminalRecovered;

    public AgentRuntimeMetrics(ObjectProvider<MeterRegistry> registries) {
        this(registries.getIfAvailable(SimpleMeterRegistry::new));
    }

    public AgentRuntimeMetrics(MeterRegistry registry) {
        MeterRegistry meters = Objects.requireNonNull(
                registry, "registry must not be null");
        Gauge.builder(PREFIX + ".runs.active", activeRuns, AtomicLong::get)
                .register(meters);
        Gauge.builder(PREFIX + ".runs.waiting", waitingRuns, AtomicLong::get)
                .register(meters);
        Gauge.builder(PREFIX + ".harness.leases.active",
                        activeHarnessLeases, AtomicLong::get)
                .register(meters);
        Gauge.builder(PREFIX + ".outbox.backlog", outboxBacklog, AtomicLong::get)
                .register(meters);
        ownerLost = counter(meters, ".runs.owner_lost");
        terminals = terminalCounters(meters, ".runs.terminal");
        harnessHit = taggedCounter(meters, ".harness.cache", "result", "hit");
        harnessMiss = taggedCounter(meters, ".harness.cache", "result", "miss");
        harnessEviction = counter(meters, ".harness.eviction");
        harnessCapacityRejected = counter(meters, ".harness.capacity_rejected");
        stateLatency = Timer.builder(PREFIX + ".state.latency").register(meters);
        stateFailure = counter(meters, ".state.failure");
        stateBulkheadRejected = counter(meters, ".state.bulkhead_rejected");
        eventPersistLatency = Timer.builder(PREFIX + ".event.persist.latency")
                .register(meters);
        eventPersisted = counter(meters, ".event.persisted");
        sequenceAllocated = counter(meters, ".event.sequence_allocated");
        eventBackpressureRejected = counter(
                meters, ".event.backpressure_rejected");
        outboxRetry = counter(meters, ".outbox.retry");
        providerTerminals = terminalCounters(meters, ".provider.terminal");
        providerClosed = counter(meters, ".provider.closed");
        toolSchedulerViolation = counter(meters, ".tool.scheduler_violation");
        replayDelivered = counter(meters, ".replay.delivered");
        replayDuplicateSuppressed = counter(
                meters, ".replay.duplicate_suppressed");
        replayTerminalRecovered = counter(
                meters, ".replay.terminal_recovered");
    }

    public static AgentRuntimeMetrics noop() {
        return NOOP;
    }

    public void executionStarted() {
        activeRuns.incrementAndGet();
    }

    public void executionStopped() {
        decrement(activeRuns);
    }

    public void waitingEntered() {
        waitingRuns.incrementAndGet();
    }

    public void waitingResumed() {
        decrement(waitingRuns);
    }

    public void ownerLost() {
        ownerLost.increment();
    }

    public void terminal(AgentRunStatus status) {
        terminalCounter(terminals, status).increment();
    }

    public void harnessCacheLookup(boolean hit) {
        (hit ? harnessHit : harnessMiss).increment();
    }

    public void harnessEvicted() {
        harnessEviction.increment();
    }

    public void harnessLeaseAcquired() {
        activeHarnessLeases.incrementAndGet();
    }

    public void harnessLeaseReleased() {
        decrement(activeHarnessLeases);
    }

    public void harnessCapacityRejected() {
        harnessCapacityRejected.increment();
    }

    public void stateOperation(Duration latency, boolean succeeded) {
        stateLatency.record(requireLatency(latency));
        if (!succeeded) {
            stateFailure.increment();
        }
    }

    public void stateBulkheadRejected() {
        stateBulkheadRejected.increment();
    }

    public void eventPersisted(Duration latency) {
        eventPersistLatency.record(requireLatency(latency));
        eventPersisted.increment();
        sequenceAllocated.increment();
    }

    public void eventBackpressureRejected() {
        eventBackpressureRejected.increment();
    }

    public void outboxBacklog(long count) {
        if (count < 0) {
            throw new IllegalArgumentException("count must not be negative");
        }
        outboxBacklog.set(count);
    }

    public void outboxRetry() {
        outboxRetry.increment();
    }

    public void providerTerminal(AgentRunStatus status) {
        terminalCounter(providerTerminals, status).increment();
    }

    public void providerClosed() {
        providerClosed.increment();
    }

    public void toolSchedulerViolation() {
        toolSchedulerViolation.increment();
    }

    public void replayDelivered() {
        replayDelivered.increment();
    }

    public void replayDuplicateSuppressed() {
        replayDuplicateSuppressed.increment();
    }

    public void replayTerminalRecovered() {
        replayTerminalRecovered.increment();
    }

    private Map<AgentRunStatus, Counter> terminalCounters(
            MeterRegistry registry, String suffix) {
        Map<AgentRunStatus, Counter> counters = new EnumMap<>(AgentRunStatus.class);
        for (AgentRunStatus status : TERMINAL_STATUSES) {
            counters.put(status, taggedCounter(
                    registry, suffix, "status", status.name()));
        }
        return Map.copyOf(counters);
    }

    private Counter terminalCounter(
            Map<AgentRunStatus, Counter> counters, AgentRunStatus status) {
        if (!TERMINAL_STATUSES.contains(status)) {
            throw new IllegalArgumentException("status must be terminal");
        }
        return counters.get(status);
    }

    private Counter counter(MeterRegistry registry, String suffix) {
        return Counter.builder(PREFIX + suffix).register(registry);
    }

    private Counter taggedCounter(
            MeterRegistry registry,
            String suffix,
            String tag,
            String value) {
        return Counter.builder(PREFIX + suffix)
                .tag(tag, value)
                .register(registry);
    }

    private Duration requireLatency(Duration latency) {
        Duration safe = Objects.requireNonNull(latency, "latency must not be null");
        if (safe.isNegative()) {
            throw new IllegalArgumentException("latency must not be negative");
        }
        return safe;
    }

    private void decrement(AtomicLong gauge) {
        gauge.updateAndGet(value -> Math.max(0, value - 1));
    }
}
