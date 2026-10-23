package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.service.ai.run.model.ExecutionStopReason;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;
import reactor.test.scheduler.VirtualTimeScheduler;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;

class AgentExecutionLimitsTests {
    @Test
    void defaultsHaveNoPlatformDeadlineOrStepBudget() {
        var config = new AgentScopeV2Properties().getExecution();
        assertThat(config.getRunTimeout()).isZero();
        assertThat(config.getMaxIters()).isZero();
        assertThat(AgentExecutionLimits.frameworkSteps(config.getMaxIters())).isEqualTo(Integer.MAX_VALUE);
        assertThat(AgentExecutionLimits.deadlineFrom(Instant.now(), config.getRunTimeout()))
                .isEqualTo(AgentExecutionLimits.UNLIMITED_DEADLINE);
    }

    @Test
    void anUnlimitedExecutionHasNoDeadlineTaskAndCanStillBeCancelled() {
        var timer = VirtualTimeScheduler.create();
        try {
            var registry = new OwnedExecutionRegistry(new ObjectMapper(), 32, 8192, timer,
                    Clock.fixed(Instant.parse("2026-10-15T00:00:00Z"), ZoneOffset.UTC));
            AtomicReference<ExecutionStopReason> stopped = new AtomicReference<>();
            var execution = new AgentExecution("run", "owner", 1, 1, "session", Flux.never(),
                    reason -> Mono.fromRunnable(() -> stopped.set(reason)), () -> {});
            registry.registerAndLaunch(execution, AgentExecutionLimits.UNLIMITED_DEADLINE,
                    events -> events, Flux::then, outcome -> Mono.empty()).block();
            assertThat(timer.getScheduledTaskCount()).isZero();
            timer.advanceTimeBy(Duration.ofDays(36500));
            assertThat(registry.size()).isEqualTo(1);
            assertThat(stopped.get()).isNull();
            assertThat(registry.interruptOwned("run", "owner", 1, ExecutionStopReason.CANCEL_REQUESTED).block()).isTrue();
            assertThat(stopped.get()).isEqualTo(ExecutionStopReason.CANCEL_REQUESTED);
            assertThat(registry.size()).isZero();
            assertThat(execution.isClosed()).isTrue();
        } finally {
            timer.dispose();
        }
    }

    @Test
    void aChildCanWaitPastTheOldThirtyMinuteRunLimit() {
        StepVerifier.withVirtualTime(() -> AgentExecutionLimits.untilDeadline(
                        Mono.delay(Duration.ofDays(3)).thenReturn("完成"), AgentExecutionLimits.UNLIMITED_DEADLINE))
                .thenAwait(Duration.ofDays(3)).expectNext("完成").verifyComplete();
    }
}
