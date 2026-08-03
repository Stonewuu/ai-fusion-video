package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.service.ai.AiStreamRedisService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;

/** Redis Pub/Sub hints that always point consumers back to committed MySQL rows. */
@Service
@RequiredArgsConstructor
@Slf4j
public class AgentRunRedisSignalService {

    private final AiStreamRedisService redisService;

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

    /** Exposed so replay/live bridges can avoid a subscribe-versus-publish race. */
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
