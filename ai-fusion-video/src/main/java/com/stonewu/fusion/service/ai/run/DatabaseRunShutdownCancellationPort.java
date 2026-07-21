package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import org.springframework.stereotype.Component;
import org.springframework.transaction.support.TransactionTemplate;
import reactor.core.publisher.Mono;

import java.util.Objects;

@Component
public final class DatabaseRunShutdownCancellationPort implements RunShutdownCancellationPort {

    private final AgentRunRepository runRepository;
    private final TransactionTemplate transactions;
    private final AgentRuntimeSchedulers schedulers;

    public DatabaseRunShutdownCancellationPort(
            AgentRunRepository runRepository,
            TransactionTemplate transactions,
            AgentRuntimeSchedulers schedulers) {
        this.runRepository = Objects.requireNonNull(
                runRepository, "runRepository must not be null");
        this.transactions = Objects.requireNonNull(transactions, "transactions must not be null");
        this.schedulers = Objects.requireNonNull(schedulers, "schedulers must not be null");
    }

    @Override
    public Mono<Void> request(String runId) {
        return Mono.fromRunnable(() -> transactions.executeWithoutResult(
                        ignored -> runRepository.requestCancellation(runId)))
                .subscribeOn(schedulers.journal())
                .then();
    }
}
