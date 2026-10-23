package com.stonewu.fusion.service.ai.agentscope.message;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.run.model.AgentModelRequests;
import io.agentscope.core.agent.AgentBase;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.Msg;
import io.agentscope.core.model.*;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/**
 * 每次实际网络订阅只提交一次，不在断线、取消或恢复时自动重发。
 */
public final class TrackedAgentChatModel extends ChatModelBase implements AutoCloseable {
    private ChatModelBase initialRequest;
    private final Supplier<ChatModelBase> nextRequest;
    private final Set<ChatModelBase> active = Collections.newSetFromMap(new IdentityHashMap<>());
    private final String modelName;
    private final int contextWindow;
    private final boolean structuredOutput;
    private final boolean structuredOutputWithTools;
    private final AiModel model;
    private final AgentModelRequests requests;
    private final String fingerprint;
    private boolean closed;

    private record RejectedResponse(BusinessException failure) {
    }

    public TrackedAgentChatModel(ChatModelBase initialRequest, Supplier<ChatModelBase> nextRequest,
                                 AiModel model, AgentModelRequests requests, String fingerprint) {
        this.initialRequest = initialRequest;
        this.nextRequest = nextRequest;
        this.modelName = initialRequest.getModelName();
        this.contextWindow = initialRequest.getContextWindowSize();
        this.structuredOutput = initialRequest.supportsNativeStructuredOutput();
        this.structuredOutputWithTools = initialRequest.supportsNativeStructuredOutputWithTools();
        this.model = model;
        this.requests = requests;
        this.fingerprint = fingerprint;
    }

    @Override
    public String getModelName() {
        return modelName;
    }

    @Override
    public int getContextWindowSize() {
        return contextWindow;
    }

    @Override
    public boolean supportsNativeStructuredOutput() {
        return structuredOutput;
    }

    @Override
    public boolean supportsNativeStructuredOutputWithTools() {
        return structuredOutputWithTools;
    }

    @Override
    protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
        return Flux.deferContextual(context -> {
            Object value = context.getOrDefault(AgentBase.RUNTIME_CONTEXT_KEY, null);
            RuntimeContext runtime = value instanceof RuntimeContext typed ? typed : null;
            RejectedResponse rejected = runtime == null ? null : runtime.get(RejectedResponse.class);
            if (rejected != null) {
                return Flux.error(rejected.failure());
            }
            String id = UUID.randomUUID().toString();
            AtomicReference<ChatUsage> usage = new AtomicReference<>();
            AgentModelResponseValidation validation = new AgentModelResponseValidation();
            AtomicReference<BusinessException> invalidResponse = new AtomicReference<>();
            Mono<AgentModelRequests.Claim> admission = Mono.fromCallable(
                            () -> requests.admit(id, model, runtime, fingerprint).orElse(null))
                    .subscribeOn(Schedulers.boundedElastic())
                    .repeatWhenEmpty(repeat -> repeat.delayElements(Duration.ofMillis(250)));
            // 每轮推理和摘要独立受理。一次提交仍然使用流式传输，正文不等待入账或整轮完成。
            return Flux.using(this::acquire,
                    delegate -> Flux.usingWhen(admission,
                            claim -> delegate.stream(messages, tools, options).doOnNext(response -> {
                                if (response.getUsage() != null) {
                                    usage.set(response.getUsage());
                                }
                                validation.accept(response);
                            }).concatWith(Mono.defer(() -> {
                                try {
                                    validation.finish();
                                    return Mono.empty();
                                } catch (BusinessException invalid) {
                                    invalidResponse.set(invalid);
                                    if (runtime != null) {
                                        runtime.put(RejectedResponse.class, new RejectedResponse(invalid));
                                    }
                                    return Mono.error(invalid);
                                }
                            })),
                            claim -> recorded(() -> requests.completed(claim, usage.get())),
                            (claim, error) -> recorded(() -> {
                                if (error == invalidResponse.get()) {
                                    requests.completed(claim, usage.get());
                                } else if (AgentModelHttpFailure.status(error) != null) {
                                    requests.failed(claim);
                                } else {
                                    requests.uncertain(claim, false);
                                }
                            }),
                            claim -> recorded(() -> requests.uncertain(claim, true))),
                    this::release).subscribeOn(Schedulers.boundedElastic());
        });
    }

    private Mono<Void> recorded(Runnable action) {
        return Mono.fromRunnable(action).subscribeOn(Schedulers.boundedElastic()).then();
    }

    private synchronized ChatModelBase acquire() {
        if (closed) {
            throw new IllegalStateException("Agent model is closed");
        }
        ChatModelBase request = initialRequest;
        initialRequest = null;
        if (request == null) {
            request = nextRequest.get();
        }
        active.add(request);
        return request;
    }

    private synchronized void release(ChatModelBase request) {
        if (active.remove(request)) {
            closeRequest(request);
        }
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (initialRequest != null) {
            active.add(initialRequest);
            initialRequest = null;
        }
        RuntimeException failure = null;
        for (ChatModelBase request : List.copyOf(active)) {
            try {
                release(request);
            } catch (RuntimeException error) {
                if (failure == null) {
                    failure = error;
                } else {
                    failure.addSuppressed(error);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    private static void closeRequest(ChatModelBase request) {
        if (!(request instanceof AutoCloseable closeable)) {
            return;
        }
        try {
            closeable.close();
        } catch (Exception error) {
            throw new IllegalStateException("Failed to close model request", error);
        }
    }
}
