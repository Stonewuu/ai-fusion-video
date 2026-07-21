package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.AgentBase;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.event.AgentEventType;
import io.agentscope.core.hook.Hook;
import io.agentscope.core.hook.HookEvent;
import io.agentscope.core.hook.PreCallEvent;
import io.agentscope.core.message.Msg;
import io.agentscope.core.middleware.AgentInput;
import io.agentscope.core.middleware.MiddlewareBase;
import io.agentscope.core.state.AgentState;
import io.agentscope.core.state.AgentStateStore;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

@Component
@Slf4j
@SuppressWarnings({"deprecation", "removal"})
public final class AgentScopeShutdownRecoveryBridge implements Hook, MiddlewareBase {

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
    public <T extends HookEvent> Mono<T> onEvent(T event) {
        if (!(event instanceof PreCallEvent preCall)) {
            return Mono.just(event);
        }
        return Mono.deferContextual(contextView -> {
            RuntimeContext runtimeContext = contextView.getOrDefault(
                    AgentBase.RUNTIME_CONTEXT_KEY, null);
            if (runtimeContext == null) {
                return Mono.just(event);
            }
            AgentState state = runtimeContext.getAgentState();
            if (state == null || !state.isShutdownInterrupted()) {
                return Mono.just(event);
            }
            List<Msg> input = preCall.getInputMessages();
            int historySize = state.getContext().size();
            if (input.size() < historySize) {
                return Mono.error(new IllegalStateException(
                        "AgentScope pre-call input is shorter than the persisted session history"));
            }
            RecoveryAttempt attempt = new RecoveryAttempt(
                    runtimeContext.getUserId(), runtimeContext.getSessionId(), state);
            runtimeContext.put(RecoveryAttempt.class, attempt);
            preCall.setInputMessages(new ArrayList<>(input.subList(0, historySize)));
            return Mono.just(event);
        });
    }

    @Override
    public int priority() {
        return 0;
    }

    @Override
    public Flux<AgentEvent> onAgent(
            Agent agent,
            RuntimeContext context,
            AgentInput input,
            Function<AgentInput, Flux<AgentEvent>> next) {
        if (context == null) {
            return next.apply(input);
        }
        return Flux.usingWhen(
                Mono.just(context),
                ignored -> next.apply(input)
                        .concatMap(event -> event.getType() == AgentEventType.AGENT_END
                                ? acknowledge(context).thenReturn(event)
                                : Mono.just(event)),
                this::clear,
                (ignored, failure) -> clear(context),
                this::clear);
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
                            .doFinally(ignored -> context.put(RecoveryAttempt.class, null));
                })
                .then();
    }

    private Mono<Void> clear(RuntimeContext context) {
        return Mono.fromRunnable(() -> context.put(RecoveryAttempt.class, null));
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
