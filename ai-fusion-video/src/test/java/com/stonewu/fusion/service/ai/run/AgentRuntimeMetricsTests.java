package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.enums.ai.AgentRunStatus;
import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class AgentRuntimeMetricsTests {

    private static final Set<String> FORBIDDEN_TAGS = Set.of(
            "runId", "conversationId", "userId", "toolCallId",
            "prompt", "modelPrompt", "secret");

    @Test
    void registersAndUpdatesEveryDurableRuntimeMetric() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        AgentRuntimeMetrics metrics = new AgentRuntimeMetrics(registry);

        metrics.executionStarted();
        metrics.waitingEntered();
        metrics.ownerLost();
        metrics.terminal(AgentRunStatus.COMPLETED);
        metrics.harnessCacheLookup(true);
        metrics.harnessCacheLookup(false);
        metrics.harnessEvicted();
        metrics.harnessLeaseAcquired();
        metrics.harnessCapacityRejected();
        metrics.stateOperation(Duration.ofMillis(3), false);
        metrics.stateBulkheadRejected();
        metrics.eventPersisted(Duration.ofMillis(2));
        metrics.eventBackpressureRejected();
        metrics.outboxBacklog(7);
        metrics.outboxRetry();
        metrics.providerTerminal(AgentRunStatus.FAILED);
        metrics.providerClosed();
        metrics.toolSchedulerViolation();
        metrics.replayDelivered();
        metrics.replayDuplicateSuppressed();
        metrics.replayTerminalRecovered();

        assertThat(registry.getMeters())
                .extracting(meter -> meter.getId().getName())
                .contains(
                        name("runs.active"), name("runs.waiting"),
                        name("runs.owner_lost"), name("runs.terminal"),
                        name("harness.cache"), name("harness.eviction"),
                        name("harness.leases.active"),
                        name("harness.capacity_rejected"),
                        name("state.latency"), name("state.failure"),
                        name("state.bulkhead_rejected"),
                        name("event.persist.latency"), name("event.persisted"),
                        name("event.sequence_allocated"),
                        name("event.backpressure_rejected"),
                        name("outbox.backlog"), name("outbox.retry"),
                        name("provider.terminal"), name("provider.closed"),
                        name("tool.scheduler_violation"),
                        name("replay.delivered"),
                        name("replay.duplicate_suppressed"),
                        name("replay.terminal_recovered"));
        assertThat(registry.get(name("runs.active")).gauge().value()).isEqualTo(1);
        assertThat(registry.get(name("runs.waiting")).gauge().value()).isEqualTo(1);
        assertThat(registry.get(name("outbox.backlog")).gauge().value()).isEqualTo(7);
        assertThat(registry.get(name("event.persisted")).counter().count())
                .isEqualTo(1);
        assertThat(registry.get(name("state.failure")).counter().count())
                .isEqualTo(1);
    }

    @Test
    void exposesNoHighCardinalityIdentityTags() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        new AgentRuntimeMetrics(registry);

        assertThat(registry.getMeters()).allSatisfy(this::hasOnlyBoundedTags);
    }

    private void hasOnlyBoundedTags(Meter meter) {
        assertThat(meter.getId().getTags())
                .extracting(tag -> tag.getKey())
                .doesNotContainAnyElementsOf(FORBIDDEN_TAGS);
        assertThat(meter.getId().getTags())
                .extracting(tag -> tag.getKey())
                .allMatch(Set.of("status", "result")::contains);
    }

    private static String name(String suffix) {
        return AgentRuntimeMetrics.PREFIX + '.' + suffix;
    }
}
