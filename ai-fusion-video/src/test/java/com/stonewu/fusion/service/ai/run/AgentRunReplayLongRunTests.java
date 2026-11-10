package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.repository.ai.AgentEventRepository.ReplaySnapshot;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import reactor.test.StepVerifier;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.LongStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class AgentRunReplayLongRunTests {
    /** 长任务逐条消费事件时，轮询不能叠加请求栈，最后仍须交付完成回执。 */
    @Test
    void thousandsOfLivePollingCyclesStayOrderedAndReachTheTerminalWithBackpressure() {
        var repository = mock(AgentEventRepository.class);
        var signals = mock(AgentRunRedisSignalService.class);
        var schedulers = mock(AgentRuntimeSchedulers.class);
        when(schedulers.journal()).thenReturn(Schedulers.immediate());
        var snapshots = new AtomicInteger();
        var released = new AtomicBoolean();
        when(signals.wakeupsWhenSubscribed("long-run"))
                .thenReturn(Mono.just(Flux.<Long>never().doFinally(ignored -> released.set(true))));
        when(repository.loadReplaySnapshot("long-run")).thenAnswer(ignored -> {
            long current = Math.min(5_000, snapshots.incrementAndGet());
            return new ReplaySnapshot(current, current == 5_000 ? current : null);
        });
        when(repository.loadReplayPage(eq("long-run"), anyLong(), anyLong(), anyInt())).thenAnswer(call ->
                LongStream.rangeClosed((long) call.getArgument(1) + 1, call.getArgument(2))
                        .mapToObj(AgentRunReplayLongRunTests::event).toList());
        var service = new AgentRunReplayService(repository, signals, schedulers);

        StepVerifier.withVirtualTime(() -> service.replayThenLive("long-run", 0).limitRate(1))
                .recordWith(ArrayList::new)
                .thenAwait(Duration.ofMinutes(30))
                .expectNextCount(5_000)
                .consumeRecordedWith(events -> assertThat(events).extracting(CommittedAgentEvent::sequence)
                        .containsExactlyElementsOf(LongStream.rangeClosed(1, 5_000).boxed().toList()))
                .verifyComplete();
        assertThat(released).isTrue();
    }

    private static CommittedAgentEvent event(long sequence) {
        String type = sequence == 5_000 ? "DONE" : "CONTENT";
        return new CommittedAgentEvent(sequence, "long-run", sequence, new AgentEventEnvelope("e-" + sequence,
                type, "main", "reply", null, null, null, null, type,
                JsonNodeFactory.instance.objectNode().put("content", "内容"), Instant.now()), Instant.now());
    }
}
