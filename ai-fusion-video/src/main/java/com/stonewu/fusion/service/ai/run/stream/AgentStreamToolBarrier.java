package com.stonewu.fusion.service.ai.run.stream;

import reactor.core.publisher.Mono;

import java.util.List;
import java.util.function.Function;

/** 让工具执行等待参数保存，让下一次模型请求等待工具结果保存。 */
public record AgentStreamToolBarrier(Function<List<String>, Mono<Void>> before,
                                     Function<List<String>, Mono<Void>> after) {
}
