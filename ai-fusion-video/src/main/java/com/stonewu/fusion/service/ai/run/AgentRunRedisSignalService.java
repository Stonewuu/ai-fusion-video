package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.service.ai.AiStreamRedisService;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import reactor.core.Disposable;
import reactor.core.Disposables;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Redis Pub/Sub hints that always point consumers back to committed MySQL rows.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class AgentRunRedisSignalService {

    private final AiStreamRedisService redisService;
    private final Map<String, Long> pending = new ConcurrentHashMap<>();
    private final AtomicBoolean sending = new AtomicBoolean();
    private final Disposable.Swap delivery = Disposables.swap();

    /**
     * 通知只用于唤醒读取者，不能让 Redis 等待阻塞后续日志写入、终态或子任务完成。
     */
    public synchronized void notifyCommitted(String runId, long sequence) {
        String safeRunId = requireRunId(runId);
        if (sequence <= 0) {
            throw new IllegalArgumentException("sequence must be positive");
        }
        if (pending.size() < 4096 || pending.containsKey(safeRunId)) {
            pending.merge(safeRunId, sequence, Math::max);
        }
    }

    @Scheduled(fixedDelay = 25, initialDelay = 25)
    public void flushCommitted() {
        if (pending.isEmpty() || !sending.compareAndSet(false, true)) {
            return;
        }
        Map<String, Long> batch = new LinkedHashMap<>();
        pending.entrySet().stream().limit(256).forEach(entry -> batch.put(entry.getKey(), entry.getValue()));
        batch.forEach((runId, sequence) -> pending.remove(runId, sequence));
        delivery.update(Flux.fromIterable(batch.entrySet())
                .flatMap(entry -> Mono.defer(() -> publishWakeup(entry.getKey(), entry.getValue()))
                        .timeout(Duration.ofSeconds(1))
                        .onErrorResume(failure -> {
                            log.debug("Agent 实时通知暂未发送，已提交日志仍可接续读取：runId={}", entry.getKey());
                            return Mono.empty();
                        }), 16)
                .doFinally(ignored -> sending.set(false))
                .subscribe());
    }

    @PreDestroy
    void closeNotifications() {
        delivery.dispose();
        pending.clear();
    }

    public Mono<Void> publishWakeup(String runId, long sequence) {
        String safeRunId = requireRunId(runId);
        if (sequence <= 0) {
            return Mono.error(new IllegalArgumentException(
                    "sequence must be positive"));
        }
        return redisService.publishRunWakeup(safeRunId, sequence).then();
    }

    public Flux<Long> wakeups(String runId) {
        return wakeupsWhenSubscribed(runId).flatMapMany(messages -> messages);
    }

    public Mono<Void> publishCancel(String runId) {
        String safeRunId = requireRunId(runId);
        return redisService.publishRunCancel(safeRunId).then();
    }

    public Flux<String> cancellations(String runId) {
        String safeRunId = requireRunId(runId);
        return redisService.runCancelPayloadsWhenSubscribed(safeRunId)
                .flatMapMany(messages -> messages)
                .filter(safeRunId::equals);
    }

    /**
     * Exposed so replay/live bridges can avoid a subscribe-versus-publish race.
     */
    public Mono<Flux<Long>> wakeupsWhenSubscribed(String runId) {
        String safeRunId = requireRunId(runId);
        return redisService.runWakeupPayloadsWhenSubscribed(safeRunId)
                .map(payloads -> payloads.handle((payload, sink) -> {
                    try {
                        long sequence = Long.parseLong(payload);
                        if (sequence > 0) {
                            sink.next(sequence);
                        } else {
                            log.warn(
                                    "Ignoring non-positive Agent run wake-up: runId={}",
                                    safeRunId);
                        }
                    } catch (NumberFormatException invalid) {
                        log.warn(
                                "Ignoring malformed Agent run wake-up: runId={}",
                                safeRunId);
                    }
                }));
    }

    private String requireRunId(String runId) {
        if (runId == null || runId.isBlank()) {
            throw new IllegalArgumentException("runId must not be blank");
        }
        if (runId.length() > 64
                || !StandardCharsets.US_ASCII.newEncoder().canEncode(runId)) {
            throw new IllegalArgumentException(
                    "runId must be at most 64 ASCII characters");
        }
        return runId;
    }
}
