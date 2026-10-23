package com.stonewu.fusion.service.ai.agentscope.message;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.media.AgentMediaInputResolver;
import io.agentscope.core.agent.AgentBase;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.Msg;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public final class MediaResolvingChatModel extends ChatModelBase implements AutoCloseable {
    private final ChatModelBase delegate;
    private final AiModel model;
    private final AgentMediaInputResolver media;
    private final AtomicBoolean closed = new AtomicBoolean();

    public MediaResolvingChatModel(ChatModelBase delegate, AiModel model, AgentMediaInputResolver media) {
        this.delegate = delegate;
        this.model = model;
        this.media = media;
    }

    @Override
    public String getModelName() {
        return delegate.getModelName();
    }

    @Override
    public int getContextWindowSize() {
        return delegate.getContextWindowSize();
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
    protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
        return Flux.deferContextual(context -> {
            if (!media.needed(messages)) {
                return delegate.stream(messages, tools, options);
            }
            Object value = context.getOrDefault(AgentBase.RUNTIME_CONTEXT_KEY, null);
            RuntimeContext runtime = value instanceof RuntimeContext typed ? typed : null;
            return Mono.fromCallable(() -> media.resolve(messages, runtime, model))
                    .subscribeOn(Schedulers.boundedElastic())
                    .flatMapMany(resolved -> delegate.stream(resolved, tools, options));
        });
    }

    @Override
    public void close() throws Exception {
        if (closed.compareAndSet(false, true) && delegate instanceof AutoCloseable closeable) {
            closeable.close();
        }
    }
}
