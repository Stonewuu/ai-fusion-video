package com.stonewu.fusion.controller.ai;

import org.springframework.http.codec.ServerSentEvent;
import reactor.core.publisher.Flux;

import java.time.Duration;

/**
 * 模型思考和工具执行期间维持 SSE 连接；心跳没有业务编号，也不改变运行状态。
 */
final class AgentPipelineStreamSupport {
    private AgentPipelineStreamSupport() {
    }

    static <T> Flux<ServerSentEvent<T>> keepAlive(Flux<ServerSentEvent<T>> source) {
        return source.publish(events -> Flux.merge(events,
                Flux.interval(Duration.ofSeconds(15))
                        .map(tick -> ServerSentEvent.<T>builder().comment("keep-alive").build())
                        .takeUntilOther(events.ignoreElements())));
    }
}
