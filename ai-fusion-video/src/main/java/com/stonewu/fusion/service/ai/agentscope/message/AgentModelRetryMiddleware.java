package com.stonewu.fusion.service.ai.agentscope.message;

import com.stonewu.fusion.service.ai.agentscope.context.CancellationContext;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreFailureGuard;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.event.CustomEvent;
import io.agentscope.core.event.ModelCallEndEvent;
import io.agentscope.core.event.ModelCallStartEvent;
import io.agentscope.core.middleware.AgentInput;
import io.agentscope.core.middleware.MiddlewareBase;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.Function;

/** 在同一运行和会话内续用 SDK 已保存的完整上下文，不重放输入或成功的工具调用。 */
@Slf4j
public final class AgentModelRetryMiddleware implements MiddlewareBase {
    public static final String EVENT_NAME = "model_retry";
    private final StateStoreFailureGuard failures;
    private final List<Duration> delays;

    public AgentModelRetryMiddleware(StateStoreFailureGuard failures) {
        this(failures, List.of(Duration.ofSeconds(2), Duration.ofSeconds(5), Duration.ofSeconds(10)));
    }

    AgentModelRetryMiddleware(StateStoreFailureGuard failures, List<Duration> delays) {
        this.failures = failures;
        this.delays = List.copyOf(delays);
    }

    @Override
    public int order() {
        return 100;
    }

    @Override
    public Flux<AgentEvent> onAgent(Agent agent, RuntimeContext context, AgentInput input,
                                    Function<AgentInput, Flux<AgentEvent>> next) {
        return Flux.defer(() -> {
            Attempt state = new Attempt(input);
            return Flux.defer(() -> {
                state.retry = false;
                return attempt(context, next, state);
            }).repeat(() -> state.retry);
        });
    }

    private Flux<AgentEvent> attempt(RuntimeContext context,
                                   Function<AgentInput, Flux<AgentEvent>> next, Attempt state) {
        return Flux.defer(() -> next.apply(state.input)).doOnNext(event -> {
            if (event instanceof ModelCallStartEvent start) {
                state.replyId = start.getReplyId();
            } else if (event instanceof ModelCallEndEvent) {
                state.retries = 0;
                state.replyId = null;
            }
        }).onErrorResume(AgentModelCallFailure.class, failure -> {
            if (!failure.retryable()) {
                return Flux.error(failure);
            }
            if (state.retries >= delays.size()) {
                return Flux.error(new AgentModelRetryExhausted(failure.exhaustedMessage(), failure));
            }
            // SDK 在抛出模型异常前保存安全状态；保存失败时停止，不能拿旧上下文重复执行。
            failures.throwIfFailed(new StateStoreSlot(context.getUserId(), context.getSessionId()));
            Duration delay = failure.delay(delays.get(state.retries++));
            var value = new LinkedHashMap<String, Object>();
            if (state.replyId != null) {
                value.put("replyId", state.replyId);
            }
            value.put("content", "模型请求暂时中断，" + delay.toSeconds() + " 秒后重试（" + state.retries + "/" + delays.size() + "）");
            log.warn("模型请求暂时失败，将在同一会话重试: attempt={}, delayMs={}, httpStatus={}, causeType={}",
                    state.retries, delay.toMillis(), AgentModelHttpFailure.status(failure),
                    failure.getCause().getClass().getSimpleName());
            state.input = new AgentInput(List.of());
            state.retry = true;
            return Flux.concat(Flux.just(new CustomEvent(EVENT_NAME, value)),
                    Mono.delay(delay).then(Mono.defer(() -> context.get(CancellationContext.class).checkpoint()))
                            .thenMany(Flux.empty()));
        });
    }

    private static final class Attempt {
        AgentInput input;
        boolean retry;
        int retries;
        String replyId;

        Attempt(AgentInput input) {
            this.input = input;
        }
    }

    public static final class AgentModelRetryExhausted extends RuntimeException {
        public AgentModelRetryExhausted(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
