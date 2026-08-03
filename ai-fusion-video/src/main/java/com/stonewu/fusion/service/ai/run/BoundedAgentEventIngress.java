package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;

import java.util.Objects;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/** Per-run count-and-byte bounded bridge between AgentScope and the durable journal. */
public final class BoundedAgentEventIngress {

    public static final int DEFAULT_MAX_EVENTS = 4096;
    public static final long DEFAULT_MAX_BYTES = 8L * 1024L * 1024L;

    private final ObjectMapper objectMapper;
    private final int maxEvents;
    private final long maxBytes;
    private final Sinks.Many<SizedEvent> sink;
    private final AtomicInteger queuedEvents = new AtomicInteger();
    private final AtomicLong queuedBytes = new AtomicLong();
    private final AtomicBoolean terminated = new AtomicBoolean();
    private final AtomicBoolean subscribed = new AtomicBoolean();

    public BoundedAgentEventIngress(ObjectMapper objectMapper, int maxEvents, long maxBytes) {
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
        if (maxEvents <= 0) {
            throw new IllegalArgumentException("maxEvents must be greater than zero");
        }
        if (maxBytes <= 0) {
            throw new IllegalArgumentException("maxBytes must be greater than zero");
        }
        this.maxEvents = maxEvents;
        this.maxBytes = maxBytes;
        this.sink = Sinks.many().unicast()
                .onBackpressureBuffer(new ArrayBlockingQueue<>(maxEvents));
    }

    public boolean offer(AgentEventEnvelope event) {
        AgentEventEnvelope safeEvent = Objects.requireNonNull(event, "event must not be null");
        if (terminated.get()) {
            return false;
        }
        long bytes = serializedBytes(safeEvent);
        if (bytes > maxBytes || !reserve(bytes)) {
            return false;
        }
        Sinks.EmitResult result = sink.tryEmitNext(new SizedEvent(safeEvent, bytes));
        if (result.isSuccess()) {
            return true;
        }
        release(bytes);
        return false;
    }

    public Flux<AgentEventEnvelope> events() {
        return Flux.defer(() -> {
            if (!subscribed.compareAndSet(false, true)) {
                return Flux.error(new IllegalStateException(
                        "Agent event ingress supports exactly one durable consumer"));
            }
            return sink.asFlux()
                    .map(sized -> {
                        release(sized.bytes());
                        return sized.event();
                    })
                    .doFinally(ignored -> resetAccounting());
        });
    }

    public void complete() {
        if (terminated.compareAndSet(false, true)) {
            sink.tryEmitComplete();
        }
    }

    public int queuedEvents() {
        return queuedEvents.get();
    }

    public long queuedBytes() {
        return queuedBytes.get();
    }

    private boolean reserve(long bytes) {
        int eventCount = queuedEvents.incrementAndGet();
        if (eventCount > maxEvents) {
            queuedEvents.decrementAndGet();
            return false;
        }
        long byteCount = queuedBytes.addAndGet(bytes);
        if (byteCount > maxBytes) {
            queuedBytes.addAndGet(-bytes);
            queuedEvents.decrementAndGet();
            return false;
        }
        return true;
    }

    private void release(long bytes) {
        queuedBytes.updateAndGet(current -> Math.max(0L, current - bytes));
        queuedEvents.updateAndGet(current -> Math.max(0, current - 1));
    }

    private void resetAccounting() {
        queuedEvents.set(0);
        queuedBytes.set(0L);
    }

    private long serializedBytes(AgentEventEnvelope event) {
        long bytes = event.payloadJson(objectMapper).getBytes(StandardCharsets.UTF_8).length;
        bytes += utf8(event.rawEventId());
        bytes += utf8(event.rawEventType());
        bytes += utf8(event.source());
        bytes += utf8(event.replyId());
        bytes += utf8(event.blockId());
        bytes += utf8(event.toolCallId());
        bytes += utf8(event.parentToolCallId());
        bytes += utf8(event.agentName());
        bytes += utf8(event.outputType());
        bytes += utf8(event.createdAt().toString());
        return bytes;
    }

    private long utf8(String value) {
        return value == null ? 0L : value.getBytes(StandardCharsets.UTF_8).length;
    }

    private record SizedEvent(AgentEventEnvelope event, long bytes) {
    }
}
