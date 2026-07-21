package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.ExecutionStopReason;
import reactor.core.Disposable;
import reactor.core.Disposables;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.util.Objects;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;

public final class AgentExecutionHandle implements AutoCloseable {

    private final AgentExecution execution;
    private final BoundedAgentEventIngress ingress;
    private final Instant deadline;
    private final Scheduler deadlineScheduler;
    private final Clock clock;
    private final Runnable removalAction;
    private final Disposable.Swap sourceSubscription = Disposables.swap();
    private final Disposable.Swap drainSubscription = Disposables.swap();
    private final Disposable.Swap deadlineSubscription = Disposables.swap();
    private final AtomicReference<Outcome> outcome = new AtomicReference<>();
    private final AtomicBoolean launched = new AtomicBoolean();
    private final AtomicBoolean closed = new AtomicBoolean();

    AgentExecutionHandle(
            AgentExecution execution,
            BoundedAgentEventIngress ingress,
            Instant deadline,
            Scheduler deadlineScheduler,
            Clock clock,
            Runnable removalAction) {
        this.execution = Objects.requireNonNull(execution, "execution must not be null");
        this.ingress = Objects.requireNonNull(ingress, "ingress must not be null");
        this.deadline = Objects.requireNonNull(deadline, "deadline must not be null");
        this.deadlineScheduler = Objects.requireNonNull(
                deadlineScheduler, "deadlineScheduler must not be null");
        this.clock = Objects.requireNonNull(clock, "clock must not be null");
        this.removalAction = Objects.requireNonNull(removalAction, "removalAction must not be null");
    }

    public String runId() {
        return execution.runId();
    }

    public String ownerInstanceId() {
        return execution.ownerInstanceId();
    }

    public long ownerEpoch() {
        return execution.ownerEpoch();
    }

    public long userId() {
        return execution.userId();
    }

    public String stateSessionId() {
        return execution.stateSessionId();
    }

    public boolean isLaunched() {
        return launched.get();
    }

    void launch(
            Function<Flux<AgentEventEnvelope>, Flux<AgentEventEnvelope>> ingressTransform,
            Function<Flux<AgentEventEnvelope>, Mono<Void>> durableConsumer,
            Function<Outcome, Mono<Void>> completionAction) {
        Objects.requireNonNull(ingressTransform, "ingressTransform must not be null");
        Objects.requireNonNull(durableConsumer, "durableConsumer must not be null");
        Objects.requireNonNull(completionAction, "completionAction must not be null");
        if (!launched.compareAndSet(false, true)) {
            throw new IllegalStateException("Agent execution handle was already launched");
        }

        Mono<Void> drain = Objects.requireNonNull(
                        durableConsumer.apply(ingress.events()),
                        "durableConsumer returned null")
                .onErrorResume(failure -> {
                    overrideWithJournalFailure(failure);
                    stopSource();
                    ingress.complete();
                    return Mono.empty();
                })
                .then(Mono.defer(() -> Objects.requireNonNull(
                        completionAction.apply(finalOutcome()),
                        "completionAction returned null")))
                .doFinally(ignored -> close());
        drainSubscription.update(drain.subscribe(
                ignored -> { },
                ignored -> { }));

        Flux<AgentEventEnvelope> transformed = Objects.requireNonNull(
                ingressTransform.apply(execution.events()),
                "ingressTransform returned null");
        sourceSubscription.update(transformed.subscribe(
                this::accept,
                this::sourceFailed,
                this::sourceCompleted));
        scheduleDeadline();
    }

    public Mono<Boolean> interrupt(ExecutionStopReason reason) {
        Objects.requireNonNull(reason, "reason must not be null");
        return Mono.defer(() -> {
            if (!outcome.compareAndSet(null, Outcome.interrupted(reason))) {
                return Mono.just(false);
            }
            stopSource();
            return execution.interrupt(reason)
                    .onErrorResume(ignored -> Mono.empty())
                    .doFinally(ignored -> ingress.complete())
                    .thenReturn(true);
        });
    }

    private void accept(AgentEventEnvelope event) {
        if (outcome.get() != null) {
            return;
        }
        if (!ingress.offer(event)
                && outcome.compareAndSet(null, Outcome.overflow())) {
            stopSource();
            ingress.complete();
        }
    }

    private void sourceCompleted() {
        outcome.compareAndSet(null, Outcome.completed());
        ingress.complete();
    }

    private void sourceFailed(Throwable failure) {
        outcome.compareAndSet(null, Outcome.sourceFailure(failure));
        ingress.complete();
    }

    private void overrideWithJournalFailure(Throwable failure) {
        outcome.updateAndGet(current -> current == null || current.kind() == OutcomeKind.COMPLETED
                ? Outcome.journalFailure(failure)
                : current);
    }

    private Outcome finalOutcome() {
        Outcome value = outcome.get();
        return value != null
                ? value
                : Outcome.journalFailure(new IllegalStateException(
                        "Agent execution ended without a terminal outcome"));
    }

    private void scheduleDeadline() {
        Duration delay = Duration.between(clock.instant(), deadline);
        if (delay.isNegative() || delay.isZero()) {
            interrupt(ExecutionStopReason.DEADLINE).subscribe();
            return;
        }
        deadlineSubscription.update(Mono.delay(delay, deadlineScheduler)
                .flatMap(ignored -> interrupt(ExecutionStopReason.DEADLINE))
                .subscribe(ignored -> { }, ignored -> { }));
    }

    private void stopSource() {
        sourceSubscription.dispose();
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        deadlineSubscription.dispose();
        sourceSubscription.dispose();
        drainSubscription.dispose();
        try {
            execution.close();
        } finally {
            removalAction.run();
        }
    }

    public enum OutcomeKind {
        COMPLETED,
        SOURCE_FAILURE,
        JOURNAL_FAILURE,
        OVERFLOW,
        INTERRUPTED
    }

    public record Outcome(
            OutcomeKind kind,
            Throwable failure,
            ExecutionStopReason stopReason) {

        public Outcome {
            Objects.requireNonNull(kind, "kind must not be null");
            if ((kind == OutcomeKind.SOURCE_FAILURE || kind == OutcomeKind.JOURNAL_FAILURE)
                    != (failure != null)) {
                throw new IllegalArgumentException("failure must match failure outcome kind");
            }
            if ((kind == OutcomeKind.INTERRUPTED) != (stopReason != null)) {
                throw new IllegalArgumentException("stopReason must match interrupted outcome");
            }
        }

        static Outcome completed() {
            return new Outcome(OutcomeKind.COMPLETED, null, null);
        }

        static Outcome sourceFailure(Throwable failure) {
            return new Outcome(OutcomeKind.SOURCE_FAILURE,
                    Objects.requireNonNull(failure, "failure must not be null"), null);
        }

        static Outcome journalFailure(Throwable failure) {
            return new Outcome(OutcomeKind.JOURNAL_FAILURE,
                    Objects.requireNonNull(failure, "failure must not be null"), null);
        }

        static Outcome overflow() {
            return new Outcome(OutcomeKind.OVERFLOW, null, null);
        }

        static Outcome interrupted(ExecutionStopReason reason) {
            return new Outcome(OutcomeKind.INTERRUPTED, null,
                    Objects.requireNonNull(reason, "reason must not be null"));
        }
    }
}
