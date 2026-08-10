package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.permission.AgentToolPermissionPolicy;
import com.stonewu.fusion.service.ai.agentscope.state.AgentStatePreflight;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.message.Msg;
import io.agentscope.harness.agent.HarnessAgent;
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
    private final AgentSessionSerialGate sessionGate;
    private final Scheduler modelScheduler;

    @Autowired
    public DefaultAgentScopeHarnessInvoker(
            HarnessLeaseCache cache,
            AgentStatePreflight preflight,
            AgentSessionSerialGate sessionGate,
            AgentRuntimeSchedulers schedulers) {
        this(cache, preflight, sessionGate,
                Objects.requireNonNull(schedulers, "schedulers must not be null").modelBlocking());
    }

    DefaultAgentScopeHarnessInvoker(
            HarnessLeaseCache cache,
            AgentStatePreflight preflight,
            Scheduler modelScheduler) {
        this(cache, preflight, new AgentSessionSerialGate(), modelScheduler);
    }

    DefaultAgentScopeHarnessInvoker(
            HarnessLeaseCache cache,
            AgentStatePreflight preflight,
            AgentSessionSerialGate sessionGate,
            Scheduler modelScheduler) {
        this.cache = Objects.requireNonNull(cache, "cache must not be null");
        this.preflight = Objects.requireNonNull(preflight, "preflight must not be null");
        this.sessionGate = Objects.requireNonNull(sessionGate, "sessionGate must not be null");
        this.modelScheduler = Objects.requireNonNull(modelScheduler, "modelScheduler must not be null");
    }

    @Override
    public Mono<Msg> call(
            AgentKernelSpec spec, List<Msg> messages, RuntimeContext context) {
        List<Msg> safeMessages = List.copyOf(Objects.requireNonNull(messages, "messages must not be null"));
        Objects.requireNonNull(context, "context must not be null");
        AgentKernelSpec safeSpec = Objects.requireNonNull(spec, "spec must not be null");
        return sessionGate.mono(context, () -> Mono.usingWhen(
                cache.acquire(safeSpec),
                lease -> Mono.defer(() -> preflight.check(context))
                        .then(Mono.defer(() ->
                                lease.resource().agent().call(safeMessages, context))),
                this::cleanup,
                (lease, failure) -> cleanup(lease),
                this::cleanup));
    }

    @Override
    public Flux<AgentEvent> streamEvents(
            AgentKernelSpec spec, List<Msg> messages, RuntimeContext context) {
        List<Msg> safeMessages = List.copyOf(Objects.requireNonNull(messages, "messages must not be null"));
        Objects.requireNonNull(context, "context must not be null");
        AgentKernelSpec safeSpec = Objects.requireNonNull(spec, "spec must not be null");
        return sessionGate.flux(context, () -> Flux.usingWhen(
                cache.acquire(safeSpec),
                lease -> Mono.defer(() -> preflight.check(context))
                        .thenMany(Flux.defer(() ->
                                streamWithPolicy(lease.resource().agent(), safeMessages, context))),
                this::cleanup,
                (lease, failure) -> cleanup(lease),
                this::cleanup));
    }

    private Flux<AgentEvent> streamWithPolicy(
            HarnessAgent agent,
            List<Msg> messages,
            RuntimeContext context) {
        AgentToolPermissionPolicy.applyRequestedPolicy(agent, context);
        return agent.streamEvents(messages, context);
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
