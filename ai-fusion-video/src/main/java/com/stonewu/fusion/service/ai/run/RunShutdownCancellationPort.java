package com.stonewu.fusion.service.ai.run;

import reactor.core.publisher.Mono;

@FunctionalInterface
public interface RunShutdownCancellationPort {
    Mono<Void> request(String runId);
}
