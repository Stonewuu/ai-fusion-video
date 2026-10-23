package com.stonewu.fusion.service.canvas;

import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasEvent;
import com.stonewu.fusion.service.canvas.model.CanvasEventPage;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 数据库事件日志是唯一事实来源；订阅不占用等待线程，也不依赖易丢失的内存广播。
 */
@Component
@Slf4j
public class CanvasEventStream {
    private static final Duration POLL_INTERVAL = Duration.ofSeconds(1);
    private static final long HEARTBEAT_NANOS = Duration.ofSeconds(15).toNanos();
    private final CanvasQueryService queries;
    private final Scheduler journal = Schedulers.newBoundedElastic(4, 2000, "canvas-event-read");

    public CanvasEventStream(CanvasQueryService queries) {
        this.queries = queries;
    }

    public Flux<ServerSentEvent<Object>> open(String canvasId, CanvasActor actor, long afterSequence) {
        // 在提交 SSE 响应之前返回授权/游标错误，连接建立后每次读取再次校验权限。
        queries.events(canvasId, actor, afterSequence, 1);
        return Flux.defer(() -> {
            AtomicLong cursor = new AtomicLong(afterSequence);
            AtomicLong heartbeat = new AtomicLong(System.nanoTime());
            return Flux.defer(() -> Mono.fromCallable(() -> queries.events(canvasId, actor, cursor.get(), 200))
                            .subscribeOn(journal)
                            .flatMapMany(page -> batch(page, cursor, heartbeat)))
                    .repeat()
                    .onErrorResume(CanvasException.class, failure -> Flux.just(ServerSentEvent.<Object>builder()
                            .event("canvas.error").data(Map.of("errorCode", failure.getErrorCode(),
                                    "message", failure.getMessage(), "details", failure.getDetails())).build()))
                    .onErrorResume(failure -> {
                        log.error("画布事件流读取失败: {}", canvasId, failure);
                        return Flux.just(ServerSentEvent.<Object>builder().event("canvas.error")
                                .data(Map.of("errorCode", "INTERNAL_ERROR", "message", "事件流读取失败，请重新连接"))
                                .build());
                    });
        });
    }

    private Flux<ServerSentEvent<Object>> batch(CanvasEventPage page, AtomicLong cursor, AtomicLong heartbeat) {
        Flux<ServerSentEvent<Object>> events = Flux.fromIterable(page.events()).map(event -> {
            cursor.set(event.sequence());
            return envelope(event);
        });
        if (page.events().isEmpty() && System.nanoTime() - heartbeat.get() >= HEARTBEAT_NANOS) {
            heartbeat.set(System.nanoTime());
            events = Flux.just(ServerSentEvent.<Object>builder().comment("canvas heartbeat").build());
        }
        if (page.hasMore()) {
            return events;
        }
        return events.concatWith(Mono.delay(POLL_INTERVAL).thenMany(Flux.empty()));
    }

    private ServerSentEvent<Object> envelope(CanvasEvent event) {
        return ServerSentEvent.<Object>builder(event).id(Long.toString(event.sequence())).event(event.type()).build();
    }

    @PreDestroy
    public void close() {
        journal.dispose();
    }
}
