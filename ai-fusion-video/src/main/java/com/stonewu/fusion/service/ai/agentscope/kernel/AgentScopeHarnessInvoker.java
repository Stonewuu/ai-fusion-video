package com.stonewu.fusion.service.ai.agentscope.kernel;

import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.message.Msg;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;

public interface AgentScopeHarnessInvoker {
    Mono<Msg> call(AgentKernelSpec spec, List<Msg> messages, RuntimeContext context);

    Flux<AgentEvent> streamEvents(
            AgentKernelSpec spec, List<Msg> messages, RuntimeContext context);
}
