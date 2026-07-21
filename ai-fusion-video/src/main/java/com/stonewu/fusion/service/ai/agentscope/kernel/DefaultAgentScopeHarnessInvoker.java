package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.state.AgentStatePreflight;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.message.Msg;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.MonoSink;
import reactor.core.scheduler.Scheduler;

import java.util.List;
import java.util.Objects;

@Component
@Slf4j
public final class DefaultAgentScopeHarnessInvoker implements AgentScopeHarnessInvoker {
    private final HarnessLeaseCache cache;
    private final AgentStatePreflight preflight;
    private final Scheduler modelScheduler;

    @Autowired
    public DefaultAgentScopeHarnessInvoker(
            HarnessLeaseCache cache,
            AgentStatePreflight preflight,
            AgentRuntimeSchedulers schedulers) {
        this(cache, preflight,
                Objects.requireNonNull(schedulers, "schedulers must not be null").modelBlocking());
    }

    DefaultAgentScopeHarnessInvoker(
            HarnessLeaseCache cache,
            AgentStatePreflight preflight,
            Scheduler modelScheduler) {
        this.cache = Objects.requireNonNull(cache, "cache must not be null");
        this.preflight = Objects.requireNonNull(preflight, "preflight must not be null");
        this.modelScheduler = Objects.requireNonNull(modelScheduler, "modelScheduler must not be null");
    }

    @Override
    public Mono<Msg> call(
            AgentKernelSpec spec, List<Msg> messages, RuntimeContext context) {
        List<Msg> safeMessages = List.copyOf(Objects.requireNonNull(messages, "messages must not be null"));
        Objects.requireNonNull(context, "context must not be null");
        return Mono.usingWhen(
                cache.acquire(Objects.requireNonNull(spec, "spec must not be null")),
                lease -> Mono.defer(() -> preflight.check(context))
                        .then(Mono.defer(() ->
                                lease.resource().agent().call(safeMessages, context))),
                this::cleanup,
                (lease, failure) -> cleanup(lease),
                this::cleanup);
    }

    @Override
    public Flux<AgentEvent> streamEvents(
            AgentKernelSpec spec, List<Msg> messages, RuntimeContext context) {
        List<Msg> safeMessages = List.copyOf(Objects.requireNonNull(messages, "messages must not be null"));
        Objects.requireNonNull(context, "context must not be null");
        return Flux.usingWhen(
                cache.acquire(Objects.requireNonNull(spec, "spec must not be null")),
                lease -> Mono.defer(() -> preflight.check(context))
                        .thenMany(Flux.defer(() ->
                                lease.resource().agent().streamEvents(safeMessages, context))),
                this::cleanup,
                (lease, failure) -> cleanup(lease),
                this::cleanup);
    }

    private Mono<Void> cleanup(HarnessLease lease) {
        return Mono.create(sink -> {
            try {
                modelScheduler.schedule(() -> closeLease(lease, sink, null));
            } catch (RuntimeException schedulingFailure) {
                log.warn("Model scheduler rejected Harness lease cleanup; releasing inline: {}",
                        schedulingFailure.toString());
                closeLease(lease, sink, schedulingFailure);
            }
        });
    }

    private void closeLease(
            HarnessLease lease,
            MonoSink<Void> sink,
            RuntimeException schedulingFailure) {
        try {
            lease.close();
            sink.success();
        } catch (Throwable closeFailure) {
            if (schedulingFailure != null && schedulingFailure != closeFailure) {
                closeFailure.addSuppressed(schedulingFailure);
            }
            sink.error(closeFailure);
        }
    }
}
