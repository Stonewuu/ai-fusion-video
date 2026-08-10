package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;

/** Deletes expired AgentScope sessions before marking their conversation expired. */
@Component
public final class AgentStateRetentionCleaner {

    private final AgentConversationMapper conversations;
    private final AgentRunMapper runs;
    private final AgentStatePreflight statePreflight;
    private final AgentRuntimeSchedulers schedulers;

    public AgentStateRetentionCleaner(
            AgentConversationMapper conversations,
            AgentRunMapper runs,
            AgentStatePreflight statePreflight,
            AgentRuntimeSchedulers schedulers) {
        this.conversations = Objects.requireNonNull(
                conversations, "conversations must not be null");
        this.runs = Objects.requireNonNull(runs, "runs must not be null");
        this.statePreflight = Objects.requireNonNull(
                statePreflight, "statePreflight must not be null");
        this.schedulers = Objects.requireNonNull(
                schedulers, "schedulers must not be null");
    }

    public Mono<Long> cleanExpired(int retentionDays, int batchSize) {
        if (retentionDays <= 0 || retentionDays > 3650) {
            return Mono.error(new IllegalArgumentException(
                    "retentionDays must be between 1 and 3650"));
        }
        if (batchSize <= 0 || batchSize > 1000) {
            return Mono.error(new IllegalArgumentException(
                    "batchSize must be between 1 and 1000"));
        }
        return journal(conversations::selectDatabaseNow)
                .flatMap(now -> cleanBatches(
                        now.minusDays(retentionDays), now, batchSize, 0L));
    }

    private Mono<Long> cleanBatches(
            LocalDateTime cutoff,
            LocalDateTime expiredAt,
            int batchSize,
            long cleaned) {
        return journal(() -> List.copyOf(
                        conversations.selectAgentStateCleanupCandidates(
                                cutoff, batchSize)))
                .flatMap(candidates -> Flux.fromIterable(candidates)
                        .concatMap(conversation -> cleanConversation(
                                conversation, expiredAt))
                        .then(Mono.defer(() -> {
                            long nextCleaned = cleaned + candidates.size();
                            return candidates.size() == batchSize
                                    ? cleanBatches(
                                            cutoff,
                                            expiredAt,
                                            batchSize,
                                            nextCleaned)
                                    : Mono.just(nextCleaned);
                        })));
    }

    private Mono<Void> cleanConversation(
            AgentConversation conversation,
            LocalDateTime expiredAt) {
        return journal(() -> List.copyOf(
                        runs.selectStateSessionIdsByConversation(
                                conversation.getConversationId())))
                .flatMapMany(Flux::fromIterable)
                .concatMap(sessionId -> statePreflight.deleteSession(
                        String.valueOf(conversation.getUserId()), sessionId))
                .then(journal(() -> conversations.markAgentStateExpired(
                                conversation.getId(),
                                conversation.getAgentStateLastActiveAt(),
                                expiredAt))
                        .then());
    }

    private <T> Mono<T> journal(Callable<T> operation) {
        return Mono.fromCallable(operation).subscribeOn(schedulers.journal());
    }
}
