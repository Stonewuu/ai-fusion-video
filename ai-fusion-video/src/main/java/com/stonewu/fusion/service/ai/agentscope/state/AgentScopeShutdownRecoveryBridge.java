package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.event.AgentEventType;
import io.agentscope.core.middleware.AgentInput;
import io.agentscope.core.middleware.MiddlewareBase;
import io.agentscope.core.state.AgentState;
import io.agentscope.core.state.AgentStateStore;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.Objects;
import java.util.Optional;
import java.util.List;
import java.util.function.Function;

/** V2 middleware that makes a shutdown-interrupted state retry exactly once. */
@Component
@Slf4j
public final class AgentScopeShutdownRecoveryBridge implements MiddlewareBase {

    private static final String AGENT_STATE_KEY = "agent_state";

    private final AgentStateStore store;
    private final AgentRuntimeSchedulers schedulers;

    public AgentScopeShutdownRecoveryBridge(
            AgentStateStore store,
            AgentRuntimeSchedulers schedulers) {
        this.store = Objects.requireNonNull(store, "store must not be null");
        this.schedulers = Objects.requireNonNull(schedulers, "schedulers must not be null");
    }

    @Override
    public Flux<AgentEvent> onAgent(
            Agent agent,
            RuntimeContext context,
            AgentInput input,
            Function<AgentInput, Flux<AgentEvent>> next) {
        Objects.requireNonNull(agent, "agent must not be null");
        Objects.requireNonNull(context, "context must not be null");
        Objects.requireNonNull(input, "input must not be null");
        Objects.requireNonNull(next, "next must not be null");
        return prepare(context, input)
                .flatMapMany(prepared -> Flux.usingWhen(
                        Mono.just(context),
                        ignored -> next.apply(prepared)
                                .concatMap(event -> event.getType() == AgentEventType.AGENT_END
                                        ? acknowledge(context).thenReturn(event)
                                        : Mono.just(event)),
                        ignored -> clear(context),
                        (ignored, failure) -> clear(context),
                        ignored -> clear(context)));
    }

    private Mono<AgentInput> prepare(RuntimeContext context, AgentInput input) {
        Objects.requireNonNull(input, "input must not be null");
        String userId = requireText(context.getUserId(), "RuntimeContext.userId");
        String sessionId = requireText(context.getSessionId(), "RuntimeContext.sessionId");
        return Mono.fromCallable(() -> Objects.requireNonNull(
                        store.get(userId, sessionId, AGENT_STATE_KEY, AgentState.class),
                        "AgentStateStore.get returned null"))
                .subscribeOn(schedulers.state())
                .map(stored -> prepareFromStored(context, input, userId, sessionId, stored));
    }

    private AgentInput prepareFromStored(
            RuntimeContext context,
            AgentInput input,
            String userId,
            String sessionId,
            Optional<AgentState> stored) {
        AgentState state = stored.orElse(null);
        if (state == null || !state.isShutdownInterrupted()) {
            context.put(RecoveryPending.class, null);
            return input;
        }
        context.put(RecoveryPending.class, new RecoveryPending(userId, sessionId));
        return new AgentInput(List.of());
    }

    @Override
    public Mono<String> onSystemPrompt(
            Agent agent,
            RuntimeContext context,
            String systemPrompt) {
        Objects.requireNonNull(agent, "agent must not be null");
        Objects.requireNonNull(context, "context must not be null");
        Objects.requireNonNull(systemPrompt, "systemPrompt must not be null");
        RecoveryPending pending = context.get(RecoveryPending.class);
        if (pending == null) {
            return Mono.just(systemPrompt);
        }
        AgentState state = context.getAgentState();
        if (state == null || !state.isShutdownInterrupted()) {
            return Mono.error(new IllegalStateException(
                    "AgentScope shutdown recovery state changed before execution"));
        }
        if (!pending.userId().equals(context.getUserId())
                || !pending.sessionId().equals(context.getSessionId())) {
            return Mono.error(new IllegalStateException(
                    "AgentScope shutdown recovery slot changed before execution"));
        }
        if (context.get(RecoveryAttempt.class) == null) {
            context.put(RecoveryAttempt.class, new RecoveryAttempt(
                    pending.userId(), pending.sessionId(), state));
        }
        return Mono.just(systemPrompt);
    }

    private Mono<Void> acknowledge(RuntimeContext context) {
        return Mono.defer(() -> {
                    RecoveryAttempt attempt = context.get(RecoveryAttempt.class);
                    if (attempt == null) {
                        return Mono.empty();
                    }
                    return Mono.fromRunnable(() -> {
                                try {
                                    attempt.state().setShutdownInterrupted(false);
                                    store.save(
                                            attempt.userId(),
                                            attempt.sessionId(),
                                            AGENT_STATE_KEY,
                                            attempt.state());
                                } catch (RuntimeException failure) {
                                    attempt.state().setShutdownInterrupted(true);
                                    throw failure;
                                }
                            })
                            .subscribeOn(schedulers.state())
                            .doOnError(failure -> log.error(
                                    "Failed to acknowledge AgentScope shutdown recovery for user={} session={}",
                                    attempt.userId(), attempt.sessionId(), failure))
                            .then(Mono.fromRunnable(() ->
                                    context.put(RecoveryAttempt.class, null)));
                })
                .then();
    }

    private Mono<Void> clear(RuntimeContext context) {
        return Mono.fromRunnable(() -> {
            RecoveryAttempt attempt = context.get(RecoveryAttempt.class);
            if (attempt != null) {
                attempt.state().setShutdownInterrupted(true);
                context.put(RecoveryAttempt.class, null);
            }
            context.put(RecoveryPending.class, null);
        });
    }

    private String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }

    private record RecoveryPending(String userId, String sessionId) {
        private RecoveryPending {
            require(userId, "userId");
            require(sessionId, "sessionId");
        }

        private static void require(String value, String field) {
            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException(field + " must not be blank");
            }
        }
    }

    private record RecoveryAttempt(String userId, String sessionId, AgentState state) {
        private RecoveryAttempt {
            if (userId == null || userId.isBlank()) {
                throw new IllegalArgumentException("userId must not be blank");
            }
            if (sessionId == null || sessionId.isBlank()) {
                throw new IllegalArgumentException("sessionId must not be blank");
            }
            Objects.requireNonNull(state, "state must not be null");
        }
    }
}
