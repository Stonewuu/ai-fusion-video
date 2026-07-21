package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Scheduler;
import reactor.test.StepVerifier;
import reactor.test.scheduler.VirtualTimeScheduler;

import java.time.Duration;
import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;

class AgentEventChunkCoalescerTests {

    @Test
    void flushesContiguousTextAtFiftyMilliseconds() {
        VirtualTimeScheduler time = VirtualTimeScheduler.create();
        AgentEventChunkCoalescer coalescer = coalescer(time, 1024);
        Flux<AgentEventEnvelope> source = Flux.concat(
                Flux.just(delta("1", "reply", "block", "你"),
                        delta("2", "reply", "block", "好")),
                Flux.never());

        StepVerifier.withVirtualTime(() -> coalescer.coalesce(source), () -> time, 1)
                .expectSubscription()
                .expectNoEvent(Duration.ofMillis(49))
                .thenAwait(Duration.ofMillis(1))
                .assertNext(event -> {
                    assertThat(event.payload().path("delta").asText()).isEqualTo("你好");
                    assertThat(event.payload().path("coalescedEventCount").asInt()).isEqualTo(2);
                    assertThat(event.rawEventId()).startsWith("coalesced:");
                })
                .thenCancel()
                .verify();
    }

    @Test
    void flushesAtCharacterLimitWithoutWaiting() {
        VirtualTimeScheduler time = VirtualTimeScheduler.create();
        AgentEventChunkCoalescer coalescer = coalescer(time, 4);
        Flux<AgentEventEnvelope> source = Flux.concat(
                Flux.just(delta("1", "reply", "block", "ab"),
                        delta("2", "reply", "block", "cd")),
                Flux.never());

        StepVerifier.withVirtualTime(() -> coalescer.coalesce(source), () -> time, 1)
                .assertNext(event -> assertThat(
                        event.payload().path("delta").asText()).isEqualTo("abcd"))
                .thenCancel()
                .verify();
    }

    @Test
    void neverCrossesIdentityAndFlushesBeforeToolBoundary() {
        AgentEventChunkCoalescer coalescer = coalescer(
                VirtualTimeScheduler.create(), 1024);
        AgentEventEnvelope first = delta("1", "reply-a", "block-a", "A");
        AgentEventEnvelope second = delta("2", "reply-b", "block-b", "B");
        AgentEventEnvelope tool = tool("3");

        StepVerifier.create(coalescer.coalesce(Flux.just(first, second, tool)))
                .expectNext(first, second, tool)
                .verifyComplete();
    }

    private AgentEventChunkCoalescer coalescer(Scheduler scheduler, int maxChars) {
        return new AgentEventChunkCoalescer(
                new ObjectMapper(),
                new AgentEventChunkCoalescer.ChunkPolicy(Duration.ofMillis(50), maxChars),
                scheduler,
                32);
    }

    private AgentEventEnvelope delta(
            String id, String replyId, String blockId, String value) {
        return new AgentEventEnvelope(
                id,
                "TEXT_BLOCK_DELTA",
                "main",
                replyId,
                blockId,
                null,
                null,
                null,
                "CONTENT",
                JsonNodeFactory.instance.objectNode().put("delta", value),
                Instant.parse("2026-07-21T00:00:00Z"));
    }

    private AgentEventEnvelope tool(String id) {
        return new AgentEventEnvelope(
                id,
                "TOOL_CALL_END",
                "main",
                "reply",
                null,
                "tool-call",
                null,
                null,
                "TOOL_CALL",
                JsonNodeFactory.instance.objectNode().put("toolName", "lookup"),
                Instant.parse("2026-07-21T00:00:00Z"));
    }
}
