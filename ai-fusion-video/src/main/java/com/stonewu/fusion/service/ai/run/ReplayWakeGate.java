package com.stonewu.fusion.service.ai.run;

import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;

import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/** Coalesces lossy Redis hints without allowing them to become replay state. */
final class ReplayWakeGate {

    private final AtomicBoolean dirty = new AtomicBoolean();
    private final AtomicLong highestHint = new AtomicLong();
    private final AtomicReference<Sinks.One<Long>> waiter = new AtomicReference<>();

    void markDirty(long sequenceHint) {
        if (sequenceHint <= 0) {
            throw new IllegalArgumentException("sequenceHint must be positive");
        }
        highestHint.accumulateAndGet(sequenceHint, Math::max);
        dirty.set(true);
        Sinks.One<Long> waiting = waiter.getAndSet(null);
        if (waiting != null) {
            waiting.tryEmitValue(sequenceHint);
        }
    }

    void requestPoll() {
        dirty.set(true);
        Sinks.One<Long> waiting = waiter.getAndSet(null);
        if (waiting != null) {
            waiting.tryEmitValue(highestHint.get());
        }
    }

    Mono<Long> awaitDirtyOrPoll(Duration pollInterval) {
        Duration safeInterval = Objects.requireNonNull(
                pollInterval, "pollInterval must not be null");
        if (safeInterval.isZero() || safeInterval.isNegative()) {
            return Mono.error(new IllegalArgumentException(
                    "pollInterval must be positive"));
        }
        return Mono.defer(() -> {
            Long ready = consumeDirty();
            if (ready != null) {
                return Mono.just(ready);
            }

            Sinks.One<Long> installed = Sinks.one();
            if (!waiter.compareAndSet(null, installed)) {
                return Mono.error(new IllegalStateException(
                        "Only one replay waiter may be active"));
            }
            ready = consumeDirty();
            if (ready != null) {
                waiter.compareAndSet(installed, null);
                installed.tryEmitEmpty();
                return Mono.just(ready);
            }

            return Mono.firstWithSignal(
                            installed.asMono(),
                            Mono.delay(safeInterval))
                    .map(ignored -> consumeAfterWake())
                    .doFinally(ignored -> waiter.compareAndSet(installed, null));
        });
    }

    private Long consumeDirty() {
        return dirty.getAndSet(false) ? highestHint.get() : null;
    }

    private long consumeAfterWake() {
        dirty.set(false);
        return highestHint.get();
    }
}
