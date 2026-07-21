package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import org.reactivestreams.Subscription;
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Autowired;
import reactor.core.Disposable;
import reactor.core.publisher.BaseSubscriber;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.HexFormat;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

/** Coalesces only identity-contiguous text/thinking deltas with a bounded output queue. */
@Component
public final class AgentEventChunkCoalescer {

    private final ObjectMapper objectMapper;
    private final ChunkPolicy policy;
    private final Scheduler timerScheduler;
    private final int maxPendingOutputs;

    @Autowired
    public AgentEventChunkCoalescer(
            ObjectMapper objectMapper,
            AgentScopeV2Properties properties) {
        this(
                objectMapper,
                new ChunkPolicy(
                        properties.getIngress().getCoalesceDelay(),
                        properties.getIngress().getCoalesceMaxChars()),
                Schedulers.parallel(),
                properties.getIngress().getMaxEvents());
    }

    AgentEventChunkCoalescer(
            ObjectMapper objectMapper,
            ChunkPolicy policy,
            Scheduler timerScheduler,
            int maxPendingOutputs) {
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
        this.policy = Objects.requireNonNull(policy, "policy must not be null");
        this.timerScheduler = Objects.requireNonNull(
                timerScheduler, "timerScheduler must not be null");
        if (maxPendingOutputs <= 0) {
            throw new IllegalArgumentException("maxPendingOutputs must be greater than zero");
        }
        this.maxPendingOutputs = maxPendingOutputs;
    }

    public Flux<AgentEventEnvelope> coalesce(Flux<AgentEventEnvelope> source) {
        Objects.requireNonNull(source, "source must not be null");
        return Flux.defer(() -> {
            Sinks.Many<AgentEventEnvelope> output = Sinks.many().unicast()
                    .onBackpressureBuffer(new ArrayBlockingQueue<>(maxPendingOutputs));
            CoalescingSubscriber subscriber = new CoalescingSubscriber(output);
            return output.asFlux()
                    .doOnSubscribe(ignored -> source.subscribe(subscriber))
                    .doFinally(ignored -> subscriber.disposeSubscriber());
        });
    }

    public record ChunkPolicy(Duration maxDelay, int maxChars) {
        public ChunkPolicy {
            Objects.requireNonNull(maxDelay, "maxDelay must not be null");
            if (maxDelay.isZero() || maxDelay.isNegative()) {
                throw new IllegalArgumentException("maxDelay must be greater than zero");
            }
            if (maxChars <= 0) {
                throw new IllegalArgumentException("maxChars must be greater than zero");
            }
        }

        public static ChunkPolicy productionDefault() {
            return new ChunkPolicy(Duration.ofMillis(50), 1024);
        }
    }

    private final class CoalescingSubscriber extends BaseSubscriber<AgentEventEnvelope> {
        private final Object monitor = new Object();
        private final Sinks.Many<AgentEventEnvelope> output;
        private DeltaAccumulator pending;
        private Disposable timer;
        private boolean done;

        private CoalescingSubscriber(Sinks.Many<AgentEventEnvelope> output) {
            this.output = output;
        }

        @Override
        protected void hookOnSubscribe(Subscription subscription) {
            request(1);
        }

        @Override
        protected void hookOnNext(AgentEventEnvelope event) {
            synchronized (monitor) {
                if (done) {
                    return;
                }
                DeltaIdentity identity = DeltaIdentity.of(event);
                String delta = delta(event);
                if (identity == null || delta == null) {
                    flushLocked();
                    emitLocked(event);
                } else if (pending == null) {
                    startLocked(event, identity, delta);
                } else if (!pending.identity().equals(identity)
                        || pending.length() + delta.length() > policy.maxChars()) {
                    flushLocked();
                    startLocked(event, identity, delta);
                } else {
                    pending.append(event, delta);
                    if (pending.length() >= policy.maxChars()) {
                        flushLocked();
                    }
                }
            }
            request(1);
        }

        @Override
        protected void hookOnComplete() {
            synchronized (monitor) {
                if (done) {
                    return;
                }
                flushLocked();
                done = true;
                output.tryEmitComplete();
            }
        }

        @Override
        protected void hookOnError(Throwable throwable) {
            synchronized (monitor) {
                if (done) {
                    return;
                }
                flushLocked();
                done = true;
                output.tryEmitError(throwable);
            }
        }

        @Override
        protected void hookOnCancel() {
            synchronized (monitor) {
                done = true;
                pending = null;
                cancelTimerLocked();
            }
        }

        private void startLocked(
                AgentEventEnvelope event, DeltaIdentity identity, String delta) {
            pending = new DeltaAccumulator(event, identity, delta);
            if (delta.length() >= policy.maxChars()) {
                flushLocked();
                return;
            }
            timer = timerScheduler.schedule(
                    this::flushFromTimer,
                    policy.maxDelay().toNanos(),
                    TimeUnit.NANOSECONDS);
        }

        private void flushFromTimer() {
            synchronized (monitor) {
                if (!done) {
                    flushLocked();
                }
            }
        }

        private void flushLocked() {
            cancelTimerLocked();
            DeltaAccumulator value = pending;
            pending = null;
            if (value != null) {
                emitLocked(value.toEnvelope());
            }
        }

        private void emitLocked(AgentEventEnvelope event) {
            Sinks.EmitResult result = output.tryEmitNext(event);
            if (result.isFailure()) {
                done = true;
                cancelTimerLocked();
                cancel();
                output.tryEmitError(new IllegalStateException(
                        "Bounded Agent event coalescer overflowed: " + result));
            }
        }

        private void cancelTimerLocked() {
            Disposable current = timer;
            timer = null;
            if (current != null) {
                current.dispose();
            }
        }

        private void disposeSubscriber() {
            synchronized (monitor) {
                done = true;
                pending = null;
                cancelTimerLocked();
            }
            cancel();
        }
    }

    private String delta(AgentEventEnvelope event) {
        JsonNode payload = event.payload();
        JsonNode value = payload.get("delta");
        return payload.isObject() && value != null && value.isTextual()
                ? value.textValue()
                : null;
    }

    private final class DeltaAccumulator {
        private final AgentEventEnvelope first;
        private final DeltaIdentity identity;
        private final StringBuilder delta;
        private final StringBuilder rawEventIds;
        private AgentEventEnvelope last;
        private int count = 1;

        private DeltaAccumulator(
                AgentEventEnvelope first, DeltaIdentity identity, String delta) {
            this.first = first;
            this.last = first;
            this.identity = identity;
            this.delta = new StringBuilder(delta);
            this.rawEventIds = new StringBuilder(first.rawEventId());
        }

        private DeltaIdentity identity() {
            return identity;
        }

        private int length() {
            return delta.length();
        }

        private void append(AgentEventEnvelope event, String value) {
            delta.append(value);
            rawEventIds.append('\0').append(event.rawEventId());
            last = event;
            count++;
        }

        private AgentEventEnvelope toEnvelope() {
            if (count == 1) {
                return first;
            }
            ObjectNode payload = (ObjectNode) first.payload();
            payload.put("delta", delta.toString());
            payload.put("coalescedEventCount", count);
            return new AgentEventEnvelope(
                    "coalesced:" + sha256(rawEventIds.toString()),
                    first.rawEventType(),
                    first.source(),
                    first.replyId(),
                    first.blockId(),
                    first.toolCallId(),
                    first.parentToolCallId(),
                    first.agentName(),
                    first.outputType(),
                    payload,
                    last.createdAt());
        }
    }

    private record DeltaIdentity(
            String rawEventType,
            String source,
            String replyId,
            String blockId,
            String parentToolCallId,
            String agentName,
            String outputType) {

        private static DeltaIdentity of(AgentEventEnvelope event) {
            boolean text = "TEXT_BLOCK_DELTA".equals(event.rawEventType())
                    && "CONTENT".equals(event.outputType());
            boolean thinking = "THINKING_BLOCK_DELTA".equals(event.rawEventType())
                    && "REASONING".equals(event.outputType());
            if (!text && !thinking) {
                return null;
            }
            return new DeltaIdentity(
                    event.rawEventType(),
                    event.source(),
                    event.replyId(),
                    event.blockId(),
                    event.parentToolCallId(),
                    event.agentName(),
                    event.outputType());
        }
    }

    private String sha256(String value) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(value.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }
}
