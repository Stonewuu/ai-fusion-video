package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.scheduler.VirtualTimeScheduler;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;

class AgentSlowJournalStreamingTests {
    @Test
    void delayedJournalWritesStillCommitIncrementallyBeforeTheModelFinishes() {
        var timer = VirtualTimeScheduler.create();
        try {
            var json = new ObjectMapper();
            var registry = new OwnedExecutionRegistry(json, 64, 65536, timer,
                    Clock.fixed(Instant.parse("2026-10-15T00:00:00Z"), ZoneOffset.UTC));
            var coalescer = new AgentEventChunkCoalescer(json,
                    new AgentEventChunkCoalescer.ChunkPolicy(Duration.ofMillis(50), 1024), timer, 64);
            AtomicInteger produced = new AtomicInteger();
            List<String> committed = new ArrayList<>();
            var source = Flux.interval(Duration.ofMillis(100), timer).take(12).map(index -> {
                produced.incrementAndGet();
                return new AgentEventEnvelope("event-" + index, "TEXT_BLOCK_DELTA", "main", "reply", "block",
                        null, null, null, "CONTENT", JsonNodeFactory.instance.objectNode().put("delta", "片段" + index), Instant.now());
            });
            var execution = new AgentExecution("run", "owner", 1, 1, "session", source, ignored -> Mono.empty(), () -> {});
            registry.registerAndLaunch(execution, AgentExecutionLimits.UNLIMITED_DEADLINE, coalescer::coalesce,
                    events -> events.concatMap(event -> Mono.delay(Duration.ofMillis(80), timer)
                            .doOnNext(ignored -> committed.add(event.rawEventId()))).then(),
                    outcome -> Mono.empty()).block();
            timer.advanceTimeBy(Duration.ofMillis(250));
            assertThat(produced).hasValue(2);
            assertThat(committed).containsExactly("event-0");
            assertThat(registry.size()).isEqualTo(1);
            timer.advanceTimeBy(Duration.ofSeconds(2));
            assertThat(committed).containsExactly("event-0", "event-1", "event-2", "event-3", "event-4", "event-5",
                    "event-6", "event-7", "event-8", "event-9", "event-10", "event-11");
            assertThat(registry.size()).isZero();
        } finally {
            timer.dispose();
        }
    }
}
