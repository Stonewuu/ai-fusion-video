package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.util.Objects;
import java.util.Optional;
import java.time.Duration;

/** Reactive adapter for owner-fenced MySQL event appends. */
@Service
@RequiredArgsConstructor
public class MySqlAgentEventJournal implements AgentEventJournal {

    private final AgentEventRepository repository;
    private final AgentRuntimeSchedulers schedulers;
    private AgentRuntimeMetrics metrics = AgentRuntimeMetrics.noop();

    @Autowired
    void setMetrics(AgentRuntimeMetrics metrics) {
        this.metrics = Objects.requireNonNull(metrics, "metrics must not be null");
    }

    @Override
    public Mono<Optional<CommittedAgentEvent>> appendOwned(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            AgentEventEnvelope event) {
        Objects.requireNonNull(event, "event must not be null");
        return Mono.defer(() -> {
            long started = System.nanoTime();
            return Mono.fromCallable(() -> repository.appendOwnedTx(
                            runId, ownerInstanceId, ownerEpoch, event))
                    .subscribeOn(schedulers.journal())
                    .doOnNext(committed -> committed.ifPresent(ignored ->
                            metrics.eventPersisted(Duration.ofNanos(
                                    Math.max(0, System.nanoTime() - started)))));
        });
    }
}
