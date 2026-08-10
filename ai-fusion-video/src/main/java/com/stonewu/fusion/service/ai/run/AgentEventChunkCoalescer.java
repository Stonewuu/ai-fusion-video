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
import java.time.Instant;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

/** Coalesces identity-contiguous streaming deltas with a bounded output queue. */
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
        private final Map<ReasoningContext, ReasoningTiming> activeReasoning = new HashMap<>();
        private final Map<ReasoningContext, ReasoningTiming> completedReasoning = new HashMap<>();
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
                event = addReasoningTiming(event);
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
            if (!identity.flushOnDelay()) {
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
                activeReasoning.clear();
                completedReasoning.clear();
                cancelTimerLocked();
            }
            cancel();
        }

        private AgentEventEnvelope addReasoningTiming(AgentEventEnvelope event) {
            ReasoningContext context = ReasoningContext.of(event);
            return switch (event.rawEventType()) {
                case "THINKING_BLOCK_START" -> {
                    ReasoningTiming previous = completedReasoning.remove(context);
                    activeReasoning.put(
                            context,
                            new ReasoningTiming(
                                    previous == null ? event.createdAt() : previous.startedAt(),
                                    null));
                    yield event;
                }
                case "THINKING_BLOCK_DELTA" -> {
                    ReasoningTiming timing = activeReasoning.computeIfAbsent(
                            context,
                            ignored -> new ReasoningTiming(event.createdAt(), null));
                    yield withReasoningTiming(event, timing.startedAt(), null);
                }
                case "THINKING_BLOCK_END" -> {
                    ReasoningTiming timing = activeReasoning.remove(context);
                    if (timing != null) {
                        completedReasoning.put(
                                context,
                                timing.completedAt(event.createdAt()));
                    }
                    yield event;
                }
                default -> attachCompletedReasoning(event, context);
            };
        }

        private AgentEventEnvelope attachCompletedReasoning(
                AgentEventEnvelope event,
                ReasoningContext context) {
            if ("TEXT_BLOCK_START".equals(event.rawEventType())) {
                ReasoningTiming active = activeReasoning.remove(context);
                if (active != null) {
                    completedReasoning.put(context, active.completedAt(event.createdAt()));
                }
                return event;
            }
            if (event.outputType() == null || "REASONING".equals(event.outputType())) {
                return event;
            }

            ReasoningTiming timing = completedReasoning.remove(context);
            if (timing == null) {
                ReasoningTiming active = activeReasoning.remove(context);
                if (active != null) {
                    timing = active.completedAt(event.createdAt());
                }
            }
            return timing == null
                    ? event
                    : withReasoningTiming(event, null, timing.durationMs());
        }
    }

    private AgentEventEnvelope withReasoningTiming(
            AgentEventEnvelope event,
            Instant startedAt,
            Long durationMs) {
        if (!event.payload().isObject()) {
            throw new IllegalArgumentException("Agent event payload must be an object");
        }
        ObjectNode payload = ((ObjectNode) event.payload()).deepCopy();
        if (startedAt != null) {
            payload.put("reasoningStartTime", startedAt.toEpochMilli());
        }
        if (durationMs != null) {
            payload.put("reasoningDurationMs", durationMs);
        }
        return new AgentEventEnvelope(
                event.rawEventId(),
                event.rawEventType(),
                event.source(),
                event.replyId(),
                event.blockId(),
                event.toolCallId(),
                event.parentToolCallId(),
                event.agentName(),
                event.outputType(),
                payload,
                event.createdAt());
    }

    private record ReasoningContext(
            String source,
            String replyId,
            String parentToolCallId,
            String agentName) {

        private static ReasoningContext of(AgentEventEnvelope event) {
            return new ReasoningContext(
                    event.source(),
                    event.replyId(),
                    event.parentToolCallId(),
                    event.agentName());
        }
    }

    private record ReasoningTiming(Instant startedAt, Long durationMs) {

        private ReasoningTiming {
            Objects.requireNonNull(startedAt, "startedAt must not be null");
        }

        private ReasoningTiming completedAt(Instant endedAt) {
            long elapsed = Math.max(0L, Duration.between(startedAt, endedAt).toMillis());
            return new ReasoningTiming(startedAt, elapsed);
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
            String toolCallId,
            String parentToolCallId,
            String agentName,
            String outputType,
            boolean flushOnDelay) {

        private static DeltaIdentity of(AgentEventEnvelope event) {
            boolean text = "TEXT_BLOCK_DELTA".equals(event.rawEventType())
                    && "CONTENT".equals(event.outputType());
            boolean thinking = "THINKING_BLOCK_DELTA".equals(event.rawEventType())
                    && "REASONING".equals(event.outputType());
            boolean toolCall = "TOOL_CALL_DELTA".equals(event.rawEventType())
                    && event.outputType() == null
                    && event.toolCallId() != null
                    && !event.toolCallId().isBlank();
            if (!text && !thinking && !toolCall) {
                return null;
            }
            return new DeltaIdentity(
                    event.rawEventType(),
                    event.source(),
                    event.replyId(),
                    event.blockId(),
                    event.toolCallId(),
                    event.parentToolCallId(),
                    event.agentName(),
                    event.outputType(),
                    !toolCall);
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
