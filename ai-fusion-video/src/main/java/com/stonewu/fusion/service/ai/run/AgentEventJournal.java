package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import reactor.core.publisher.Mono;

import java.util.Optional;

/** Owner-fenced reactive event journal. */
public interface AgentEventJournal {

    Mono<Optional<CommittedAgentEvent>> appendOwned(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            AgentEventEnvelope event);
}
