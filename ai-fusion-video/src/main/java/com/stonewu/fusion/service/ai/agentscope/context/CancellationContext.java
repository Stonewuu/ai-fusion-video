package com.stonewu.fusion.service.ai.agentscope.context;

import reactor.core.publisher.Mono;

import java.util.Objects;
import java.util.function.Supplier;

public record CancellationContext(Supplier<? extends Mono<Void>> checkpointAction) {

    public CancellationContext {
        checkpointAction = Objects.requireNonNull(checkpointAction, "checkpointAction must not be null");
    }

    public static CancellationContext noop() {
        return new CancellationContext(Mono::empty);
    }

    public Mono<Void> checkpoint() {
        return Mono.defer(() -> {
            Mono<Void> result = checkpointAction.get();
            if (result == null) {
                return Mono.error(new IllegalStateException("cancellation checkpoint returned null"));
            }
            return result;
        });
    }
}
