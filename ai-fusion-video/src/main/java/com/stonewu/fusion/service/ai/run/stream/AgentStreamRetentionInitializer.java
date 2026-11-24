package com.stonewu.fusion.service.ai.run.stream;

import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

/** 启动时为此前已经结束的缓存补齐原结束时间对应的期限，不延长已有保留时间。 */
@Component
@RequiredArgsConstructor
public class AgentStreamRetentionInitializer implements ApplicationRunner {
    private final AgentRedisStreamStore streams;
    private final AgentStreamControlPublisher controls;
    private final AgentRuntimeSchedulers schedulers;

    @Override
    public void run(ApplicationArguments arguments) {
        initialize().block();
    }

    public Mono<Void> initialize() {
        return Mono.fromCallable(streams::existingRuns).subscribeOn(schedulers.journal())
                .flatMapMany(Flux::fromIterable).concatMap(controls::synchronize).then();
    }
}
