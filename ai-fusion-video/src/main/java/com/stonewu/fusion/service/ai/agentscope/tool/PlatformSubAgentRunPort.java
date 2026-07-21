package com.stonewu.fusion.service.ai.agentscope.tool;

import reactor.core.publisher.Mono;

public interface PlatformSubAgentRunPort {

    Mono<PlatformSubAgentRun> start(PlatformSubAgentCommand command);

    Mono<Void> cancelChildren(String parentRunId);
}
