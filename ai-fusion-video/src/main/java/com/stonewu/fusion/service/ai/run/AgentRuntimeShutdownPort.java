package com.stonewu.fusion.service.ai.run;

import reactor.core.publisher.Mono;

import java.time.Duration;

public interface AgentRuntimeShutdownPort {
    Mono<Void> shutdown(Duration drainTimeout);
}
