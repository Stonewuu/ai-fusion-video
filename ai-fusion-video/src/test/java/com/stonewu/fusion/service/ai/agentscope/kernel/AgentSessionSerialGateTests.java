package com.stonewu.fusion.service.ai.agentscope.kernel;

import io.agentscope.core.agent.RuntimeContext;
import org.junit.jupiter.api.Test;
import reactor.core.Disposable;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.test.StepVerifier;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;

class AgentSessionSerialGateTests {

    private final AgentSessionSerialGate gate = new AgentSessionSerialGate();

    @Test
    void queuedCancellationDoesNotLetALaterTurnOvertakeTheActiveTurn() {
        RuntimeContext context = context("42", "conversation-7");
        Sinks.Empty<Void> releaseFirst = Sinks.empty();
        List<String> starts = new CopyOnWriteArrayList<>();
        Mono<String> first = gate.mono(context, () -> {
                    starts.add("first");
                    return releaseFirst.asMono().thenReturn("first-result");
                })
                .cache();
        Mono<String> cancelled = gate.mono(context, () -> {
                    starts.add("cancelled");
                    return Mono.just("cancelled-result");
                });
        Mono<String> last = gate.mono(context, () -> {
                    starts.add("last");
                    return Mono.just("last-result");
                })
                .cache();

        first.subscribe();
        Disposable cancelledSubscription = cancelled.subscribe();
        last.subscribe();
        assertThat(starts).containsExactly("first");

        cancelledSubscription.dispose();
        assertThat(starts).containsExactly("first");
        releaseFirst.tryEmitEmpty();

        StepVerifier.create(Mono.zip(first, last))
                .assertNext(results -> {
                    assertThat(results.getT1()).isEqualTo("first-result");
                    assertThat(results.getT2()).isEqualTo("last-result");
                })
                .verifyComplete();
        assertThat(starts).containsExactly("first", "last");
        assertThat(gate.laneCount()).isZero();
    }

    @Test
    void differentSlotsRunIndependently() {
        Sinks.Empty<Void> releaseFirst = Sinks.empty();
        List<String> starts = new CopyOnWriteArrayList<>();
        Mono<String> first = gate.mono(context("42", "conversation-a"), () -> {
                    starts.add("first");
                    return releaseFirst.asMono().thenReturn("first-result");
                })
                .cache();
        Mono<String> other = gate.mono(context("42", "conversation-b"), () -> {
                    starts.add("other");
                    return Mono.just("other-result");
                })
                .cache();

        first.subscribe();
        StepVerifier.create(other)
                .expectNext("other-result")
                .verifyComplete();
        assertThat(starts).containsExactly("first", "other");
        assertThat(gate.laneCount()).isEqualTo(1);

        releaseFirst.tryEmitEmpty();
        StepVerifier.create(first)
                .expectNext("first-result")
                .verifyComplete();
        assertThat(gate.laneCount()).isZero();
    }

    @Test
    void activeCancellationHandsTheSlotToTheNextTurnExactlyOnce() {
        RuntimeContext context = context("42", "conversation-cancel");
        AtomicInteger nextStarts = new AtomicInteger();
        Disposable active = gate.<String>mono(context, Mono::never).subscribe();
        Mono<String> next = gate.mono(context, () -> {
                    nextStarts.incrementAndGet();
                    return Mono.just("next-result");
                })
                .cache();

        next.subscribe();
        assertThat(nextStarts).hasValue(0);
        active.dispose();

        StepVerifier.create(next)
                .expectNext("next-result")
                .verifyComplete();
        assertThat(nextStarts).hasValue(1);
        assertThat(gate.laneCount()).isZero();
    }

    @Test
    void releasesLaneAfterCompletionErrorSupplierFailureAndActiveCancellation() {
        RuntimeContext context = context("42", "conversation-cleanup");

        StepVerifier.create(gate.mono(context, () -> Mono.just("done")))
                .expectNext("done")
                .verifyComplete();
        assertThat(gate.laneCount()).isZero();

        StepVerifier.create(gate.mono(
                        context, () -> Mono.error(new IllegalStateException("work failed"))))
                .expectErrorMessage("work failed")
                .verify();
        assertThat(gate.laneCount()).isZero();

        StepVerifier.create(gate.<String>mono(context, () -> {
                    throw new IllegalStateException("supplier failed");
                }))
                .expectErrorMessage("supplier failed")
                .verify();
        assertThat(gate.laneCount()).isZero();

        StepVerifier.create(gate.<String>mono(context, Mono::never))
                .thenCancel()
                .verify();
        assertThat(gate.laneCount()).isZero();
    }

    private static RuntimeContext context(String userId, String sessionId) {
        return RuntimeContext.builder()
                .userId(userId)
                .sessionId(sessionId)
                .build();
    }
}
