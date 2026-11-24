package com.stonewu.fusion.service.ai.run.stream;

import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.event.ToolResultEndEvent;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.middleware.ActingInput;
import io.agentscope.core.middleware.MiddlewareBase;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

public final class AgentStreamToolMiddleware implements MiddlewareBase {
    @Override
    public Flux<AgentEvent> onActing(Agent agent, RuntimeContext context, ActingInput input,
                                    Function<ActingInput, Flux<AgentEvent>> next) {
        AgentRunContext run = context.get(AgentRunContext.class);
        if (run == null || run.streamProtocolVersion() == 1) {
            return next.apply(input);
        }
        AgentStreamToolBarrier barrier = Objects.requireNonNull(context.get(AgentStreamToolBarrier.class),
                "消息流缺少工具保存协调器");
        List<String> calls = input.toolCalls().stream().map(ToolUseBlock::getId).toList();
        Set<String> completed = new LinkedHashSet<>();
        return barrier.before().apply(calls).thenMany(Flux.defer(() -> next.apply(input)))
                .doOnNext(event -> {
                    if (event instanceof ToolResultEndEvent end && calls.contains(end.getToolCallId())) {
                        completed.add(end.getToolCallId());
                    }
                })
                .concatWith(Mono.defer(() -> barrier.after().apply(List.copyOf(completed)))
                        .thenMany(Flux.empty()));
    }
}
