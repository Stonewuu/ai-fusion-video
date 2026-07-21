package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.util.Objects;
import java.util.Optional;

/** Reactive adapter for owner-fenced MySQL event appends. */
@Service
@RequiredArgsConstructor
public class MySqlAgentEventJournal implements AgentEventJournal {

    private final AgentEventRepository repository;
    private final AgentRuntimeSchedulers schedulers;

    @Override
    public Mono<Optional<CommittedAgentEvent>> appendOwned(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            AgentEventEnvelope event) {
        Objects.requireNonNull(event, "event must not be null");
        return Mono.fromCallable(() -> repository.appendOwnedTx(
                        runId, ownerInstanceId, ownerEpoch, event))
                .subscribeOn(schedulers.journal());
    }
}
