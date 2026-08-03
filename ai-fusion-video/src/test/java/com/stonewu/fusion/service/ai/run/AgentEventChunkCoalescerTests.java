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
import java.util.stream.IntStream;

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

    @Test
    void coalescesLargeToolArgumentsBySizeUntilTheToolBoundary() {
        AgentEventChunkCoalescer coalescer = coalescer(
                VirtualTimeScheduler.create(), 128);
        Flux<AgentEventEnvelope> fragments = Flux.fromStream(IntStream.range(0, 2000)
                .mapToObj(index -> toolDelta(String.valueOf(index), "tool-call", "x")));

        StepVerifier.create(coalescer.coalesce(Flux.concat(
                                fragments,
                                Flux.just(tool("end"))))
                        .collectList())
                .assertNext(events -> {
                    assertThat(events).hasSize(17);
                    assertThat(events.getLast().rawEventType()).isEqualTo("TOOL_CALL_END");
                    assertThat(events.subList(0, events.size() - 1))
                            .allSatisfy(event -> {
                                assertThat(event.rawEventType()).isEqualTo("TOOL_CALL_DELTA");
                                assertThat(event.toolCallId()).isEqualTo("tool-call");
                                assertThat(event.payload().path("delta").asText()).hasSizeLessThanOrEqualTo(128);
                            });
                    String reassembled = events.subList(0, events.size() - 1).stream()
                            .map(event -> event.payload().path("delta").asText())
                            .reduce("", String::concat);
                    assertThat(reassembled).isEqualTo("x".repeat(2000));
                })
                .verifyComplete();
    }

    @Test
    void persistsReasoningStartAndFinalDurationAcrossTheReplayableEvents() {
        AgentEventChunkCoalescer coalescer = coalescer(
                VirtualTimeScheduler.create(), 1024);

        StepVerifier.create(coalescer.coalesce(Flux.just(
                        blockEvent("1", "THINKING_BLOCK_START", null,
                                "2026-07-21T00:00:00Z"),
                        reasoningDelta("2", "正在思考",
                                "2026-07-21T00:00:00.200Z"),
                        blockEvent("3", "THINKING_BLOCK_END", null,
                                "2026-07-21T00:00:02.500Z"),
                        blockEvent("4", "TEXT_BLOCK_START", null,
                                "2026-07-21T00:00:02.600Z"),
                        contentDelta("5", "回答",
                                "2026-07-21T00:00:02.700Z"))))
                .assertNext(event -> assertThat(event.rawEventType())
                        .isEqualTo("THINKING_BLOCK_START"))
                .assertNext(event -> {
                    assertThat(event.rawEventType()).isEqualTo("THINKING_BLOCK_DELTA");
                    assertThat(event.payload().path("reasoningStartTime").asLong())
                            .isEqualTo(Instant.parse("2026-07-21T00:00:00Z").toEpochMilli());
                })
                .assertNext(event -> assertThat(event.rawEventType())
                        .isEqualTo("THINKING_BLOCK_END"))
                .assertNext(event -> assertThat(event.rawEventType())
                        .isEqualTo("TEXT_BLOCK_START"))
                .assertNext(event -> {
                    assertThat(event.rawEventType()).isEqualTo("TEXT_BLOCK_DELTA");
                    assertThat(event.payload().path("reasoningDurationMs").asLong())
                            .isEqualTo(2500L);
                })
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

    private AgentEventEnvelope reasoningDelta(
            String id, String value, String createdAt) {
        return new AgentEventEnvelope(
                id,
                "THINKING_BLOCK_DELTA",
                "main",
                "reply",
                "thinking-block",
                null,
                null,
                null,
                "REASONING",
                JsonNodeFactory.instance.objectNode().put("delta", value),
                Instant.parse(createdAt));
    }

    private AgentEventEnvelope contentDelta(
            String id, String value, String createdAt) {
        return new AgentEventEnvelope(
                id,
                "TEXT_BLOCK_DELTA",
                "main",
                "reply",
                "text-block",
                null,
                null,
                null,
                "CONTENT",
                JsonNodeFactory.instance.objectNode().put("delta", value),
                Instant.parse(createdAt));
    }

    private AgentEventEnvelope blockEvent(
            String id, String rawEventType, String outputType, String createdAt) {
        String blockId = rawEventType.startsWith("THINKING")
                ? "thinking-block"
                : "text-block";
        return new AgentEventEnvelope(
                id,
                rawEventType,
                "main",
                "reply",
                blockId,
                null,
                null,
                null,
                outputType,
                JsonNodeFactory.instance.objectNode(),
                Instant.parse(createdAt));
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

    private AgentEventEnvelope toolDelta(
            String id, String toolCallId, String value) {
        return new AgentEventEnvelope(
                id,
                "TOOL_CALL_DELTA",
                "main",
                "reply",
                null,
                toolCallId,
                null,
                null,
                null,
                JsonNodeFactory.instance.objectNode()
                        .put("toolCallName", "__fragment__")
                        .put("delta", value),
                Instant.parse("2026-07-21T00:00:00Z"));
    }
}
