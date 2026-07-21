package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.service.ai.run.model.ExecutionStopReason;
import com.stonewu.fusion.service.ai.run.model.ResumeAgentExecutionCommand;
import com.stonewu.fusion.service.ai.run.model.StartAgentExecutionCommand;
import reactor.core.publisher.Mono;

public interface RunExecutionSupervisor extends AgentRuntimeShutdownPort {

    Mono<Void> start(StartAgentExecutionCommand command);

    Mono<Void> resume(ResumeAgentExecutionCommand command);

    Mono<Boolean> interruptOwned(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            ExecutionStopReason reason);
}
