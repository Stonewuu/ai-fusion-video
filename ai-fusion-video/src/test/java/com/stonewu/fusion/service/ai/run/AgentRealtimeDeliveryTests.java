package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.repository.ai.AgentEventRepository.ReplaySnapshot;
import com.stonewu.fusion.service.ai.AiStreamRedisService;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import reactor.test.StepVerifier;

import java.time.Duration;
import java.time.Instant;
import java.util.List;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentRealtimeDeliveryTests {
    @Test
    void redisSubscriptionHandshakeCannotHoldBackCommittedHistory() {
        var repository = mock(AgentEventRepository.class);
        var signals = mock(AgentRunRedisSignalService.class);
        var schedulers = mock(AgentRuntimeSchedulers.class);
        when(schedulers.journal()).thenReturn(Schedulers.immediate());
        when(signals.wakeupsWhenSubscribed("run")).thenReturn(Mono.never());
        when(repository.loadReplaySnapshot("run")).thenReturn(new ReplaySnapshot(1, 1L));
        var envelope = new AgentEventEnvelope("terminal", "AGENT_END", "main", null, null, null, null, null,
                "DONE", JsonNodeFactory.instance.objectNode().put("outputType", "DONE"), Instant.now());
        var terminal = new CommittedAgentEvent(1, "run", 1, envelope, Instant.now());
        when(repository.loadReplayPage("run", 0, 1, 256)).thenReturn(List.of(terminal));
        StepVerifier.create(new AgentRunReplayService(repository, signals, schedulers).replayThenLive("run", 0))
                .expectNext(terminal).expectComplete().verify(Duration.ofSeconds(1));
    }

    @Test
    void notificationReceiptDoesNotWaitForRedisAndCoalescesByRun() {
        var redis = mock(AiStreamRedisService.class);
        when(redis.publishRunWakeup(anyString(), anyLong())).thenReturn(Mono.never());
        var signals = new AgentRunRedisSignalService(redis);
        try {
            signals.notifyCommitted("one", 1);
            signals.notifyCommitted("one", 5);
            signals.notifyCommitted("two", 2);
            verify(redis, never()).publishRunWakeup(anyString(), anyLong());
            signals.flushCommitted();
            verify(redis).publishRunWakeup("one", 5);
            verify(redis).publishRunWakeup("two", 2);
            // 上一批 Redis 永不响应，生产者仍可继续接收新的已提交序号。
            signals.notifyCommitted("one", 6);
            signals.notifyCommitted("three", 3);
        } finally {
            signals.closeNotifications();
        }
    }
}
