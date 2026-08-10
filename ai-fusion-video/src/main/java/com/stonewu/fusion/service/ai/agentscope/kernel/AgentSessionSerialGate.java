package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import io.agentscope.core.agent.RuntimeContext;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.MonoSink;

import java.util.ArrayDeque;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

@Component
public final class AgentSessionSerialGate {

    private final ConcurrentHashMap<StateStoreSlot, Lane> lanes = new ConcurrentHashMap<>();

    public <T> Mono<T> mono(RuntimeContext context, Supplier<Mono<T>> work) {
        StateStoreSlot slot = slot(context);
        Objects.requireNonNull(work, "work must not be null");
        return Mono.defer(() -> Mono.using(
                () -> retain(slot),
                turn -> turn.await().then(Mono.defer(work)),
                Turn::close,
                true));
    }

    public <T> Flux<T> flux(RuntimeContext context, Supplier<Flux<T>> work) {
        StateStoreSlot slot = slot(context);
        Objects.requireNonNull(work, "work must not be null");
        return Flux.defer(() -> Flux.using(
                () -> retain(slot),
                turn -> turn.await().thenMany(Flux.defer(work)),
                Turn::close,
                true));
    }

    int laneCount() {
        return lanes.size();
    }

    private StateStoreSlot slot(RuntimeContext context) {
        RuntimeContext safeContext = Objects.requireNonNull(context, "context must not be null");
        return new StateStoreSlot(safeContext.getUserId(), safeContext.getSessionId());
    }

    private Turn retain(StateStoreSlot slot) {
        AtomicReference<Lane> retained = new AtomicReference<>();
        lanes.compute(slot, (ignored, existing) -> {
            Lane lane = existing == null ? new Lane(slot) : existing;
            lane.references++;
            retained.set(lane);
            return lane;
        });
        return new Turn(retained.get());
    }

    private void release(Lane lane) {
        lanes.computeIfPresent(lane.slot, (ignored, existing) -> {
            if (existing != lane) {
                return existing;
            }
            lane.references--;
            if (lane.references < 0) {
                throw new IllegalStateException("AgentScope session lane reference count underflow");
            }
            return lane.references == 0 ? null : lane;
        });
    }

    private enum TurnState {
        NEW,
        WAITING,
        GRANTED,
        CLOSED
    }

    private static final class Lane {
        private final StateStoreSlot slot;
        private final ArrayDeque<Turn> waiting = new ArrayDeque<>();
        private boolean held;
        private int references;

        private Lane(StateStoreSlot slot) {
            this.slot = slot;
        }
    }

    private final class Turn implements AutoCloseable {
        private final Lane lane;
        private TurnState state = TurnState.NEW;
        private MonoSink<Void> sink;

        private Turn(Lane lane) {
            this.lane = lane;
        }

        private Mono<Void> await() {
            return Mono.create(candidate -> {
                boolean granted = false;
                IllegalStateException failure = null;
                synchronized (lane) {
                    if (state == TurnState.CLOSED) {
                        failure = new IllegalStateException(
                                "AgentScope session turn was closed before it could start");
                    } else if (state != TurnState.NEW) {
                        failure = new IllegalStateException(
                                "AgentScope session turn may only be awaited once");
                    } else {
                        sink = candidate;
                        if (lane.held) {
                            state = TurnState.WAITING;
                            lane.waiting.addLast(this);
                            granted = false;
                        } else {
                            lane.held = true;
                            state = TurnState.GRANTED;
                            granted = true;
                        }
                    }
                }
                if (failure != null) {
                    candidate.error(failure);
                } else if (granted) {
                    candidate.success();
                }
            });
        }

        @Override
        public void close() {
            MonoSink<Void> nextSink = null;
            synchronized (lane) {
                if (state == TurnState.CLOSED) {
                    return;
                }
                if (state == TurnState.WAITING) {
                    lane.waiting.remove(this);
                } else if (state == TurnState.GRANTED) {
                    Turn next = nextWaitingTurn();
                    if (next == null) {
                        lane.held = false;
                    } else {
                        next.state = TurnState.GRANTED;
                        nextSink = next.sink;
                    }
                }
                state = TurnState.CLOSED;
            }
            release(lane);
            if (nextSink != null) {
                nextSink.success();
            }
        }

        private Turn nextWaitingTurn() {
            Turn next;
            while ((next = lane.waiting.pollFirst()) != null) {
                if (next.state == TurnState.WAITING) {
                    return next;
                }
            }
            return null;
        }
    }
}
