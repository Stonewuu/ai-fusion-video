package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.common.BusinessException;
import io.agentscope.core.agent.AgentBase;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.Msg;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.Objects;

public final class StateStoreGuardedChatModel extends ChatModelBase {

    private static final String MISSING_RUNTIME_MESSAGE =
            "STATE_STORE_FAILED: RuntimeContext missing";

    private final ChatModelBase delegate;
    private final StateStoreFailureGuard failures;
    private final int contextWindowSize;

    public StateStoreGuardedChatModel(
            ChatModelBase delegate, StateStoreFailureGuard failures) {
        this(delegate, failures, null);
    }

    public StateStoreGuardedChatModel(
            ChatModelBase delegate,
            StateStoreFailureGuard failures,
            Integer configuredContextWindow) {
        this.delegate = Objects.requireNonNull(delegate, "delegate must not be null");
        this.failures = Objects.requireNonNull(failures, "failures must not be null");
        this.contextWindowSize = configuredContextWindow != null && configuredContextWindow > 0
                ? configuredContextWindow
                : delegate.getContextWindowSize();
    }

    @Override
    public String getModelName() {
        return delegate.getModelName();
    }

    @Override
    public int getContextWindowSize() {
        return contextWindowSize;
    }

    @Override
    public boolean supportsNativeStructuredOutput() {
        return delegate.supportsNativeStructuredOutput();
    }

    @Override
    public boolean supportsNativeStructuredOutputWithTools() {
        return delegate.supportsNativeStructuredOutputWithTools();
    }

    @Override
    protected Flux<ChatResponse> doStream(
            List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
        return Flux.deferContextual(contextView -> {
            Object value = contextView.getOrDefault(AgentBase.RUNTIME_CONTEXT_KEY, null);
            if (!(value instanceof RuntimeContext runtimeContext)
                    || runtimeContext.getUserId() == null
                    || runtimeContext.getUserId().isBlank()
                    || runtimeContext.getSessionId() == null
                    || runtimeContext.getSessionId().isBlank()) {
                return Flux.error(new BusinessException(500, MISSING_RUNTIME_MESSAGE));
            }
            failures.throwIfFailed(new StateStoreSlot(
                    runtimeContext.getUserId(), runtimeContext.getSessionId()));
            return delegate.stream(messages, tools, options);
        });
    }
}
