package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.stonewu.fusion.service.ai.agentscope.context.ToolPermissionContext;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.agentscope.state.AgentStatePreflight;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.message.Msg;
import io.agentscope.harness.agent.HarnessAgent;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.mockito.Answers;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;
import reactor.test.StepVerifier;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DefaultAgentScopeHarnessInvokerTests {
    private final Scheduler modelScheduler = Schedulers.newSingle("invoker-model-test");

    @AfterEach
    void closeScheduler() {
        modelScheduler.dispose();
    }

    @Test
    void callReleasesLeaseOnCompleteErrorAndCancel() {
        InvocationFixture complete = fixture();
        Msg response = mock(Msg.class);
        when(complete.agent.call(any(List.class), any(RuntimeContext.class)))
                .thenReturn(Mono.just(response));
        StepVerifier.create(complete.invoker.call(complete.spec, List.of(), complete.context))
                .expectNext(response)
                .verifyComplete();
        verify(complete.lease, timeout(1000)).close();

        InvocationFixture error = fixture();
        when(error.agent.call(any(List.class), any(RuntimeContext.class)))
                .thenReturn(Mono.error(new IllegalStateException("boom")));
        StepVerifier.create(error.invoker.call(error.spec, List.of(), error.context))
                .expectErrorMessage("boom")
                .verify();
        verify(error.lease, timeout(1000)).close();

        InvocationFixture cancel = fixture();
        when(cancel.agent.call(any(List.class), any(RuntimeContext.class)))
                .thenReturn(Mono.never());
        StepVerifier.create(cancel.invoker.call(cancel.spec, List.of(), cancel.context))
                .then(() -> verify(cancel.agent, timeout(1000))
                        .call(any(List.class), any(RuntimeContext.class)))
                .thenCancel()
                .verify();
        verify(cancel.lease, timeout(1000)).close();
    }

    @Test
    void streamEventsRunsPreflightAndReleasesLeaseOnCancel() {
        InvocationFixture fixture = fixture();
        AgentEvent event = mock(AgentEvent.class);
        when(fixture.agent.streamEvents(any(List.class), any(RuntimeContext.class)))
                .thenReturn(Flux.concat(Flux.just(event), Flux.never()));

        StepVerifier.create(fixture.invoker.streamEvents(
                        fixture.spec, List.of(), fixture.context))
                .expectNext(event)
                .thenCancel()
                .verify();

        verify(fixture.preflight, times(1)).check(fixture.context);
        verify(fixture.lease, timeout(1000).times(1)).close();
    }

    @Test
    void streamEventsReleasesLeaseOnCompleteAndError() {
        InvocationFixture complete = fixture();
        AgentEvent event = mock(AgentEvent.class);
        when(complete.agent.streamEvents(any(List.class), any(RuntimeContext.class)))
                .thenReturn(Flux.just(event));
        StepVerifier.create(complete.invoker.streamEvents(
                        complete.spec, List.of(), complete.context))
                .expectNext(event)
                .verifyComplete();
        verify(complete.lease, timeout(1000)).close();

        InvocationFixture error = fixture();
        when(error.agent.streamEvents(any(List.class), any(RuntimeContext.class)))
                .thenReturn(Flux.error(new IllegalStateException("stream failed")));
        StepVerifier.create(error.invoker.streamEvents(
                        error.spec, List.of(), error.context))
                .expectErrorMessage("stream failed")
                .verify();
        verify(error.lease, timeout(1000)).close();
    }

    @Test
    void waitsForPreflightBeforeCallingAgentAndReleasesOnPreflightFailure() {
        InvocationFixture delayed = fixture();
        Sinks.Empty<Void> preflightCompletion = Sinks.empty();
        Msg response = mock(Msg.class);
        when(delayed.preflight.check(delayed.context)).thenReturn(preflightCompletion.asMono());
        when(delayed.agent.call(any(List.class), any(RuntimeContext.class)))
                .thenReturn(Mono.just(response));

        StepVerifier.create(delayed.invoker.call(delayed.spec, List.of(), delayed.context))
                .then(() -> verify(delayed.agent, never()).call(any(List.class), any(RuntimeContext.class)))
                .then(preflightCompletion::tryEmitEmpty)
                .expectNext(response)
                .verifyComplete();
        verify(delayed.lease, timeout(1000)).close();

        InvocationFixture failed = fixture();
        when(failed.preflight.check(failed.context))
                .thenReturn(Mono.error(new IllegalStateException("preflight failed")));

        StepVerifier.create(failed.invoker.call(failed.spec, List.of(), failed.context))
                .expectErrorMessage("preflight failed")
                .verify();
        verify(failed.agent, never()).call(any(List.class), any(RuntimeContext.class));
        verify(failed.lease, timeout(1000)).close();

        InvocationFixture synchronousFailure = fixture();
        when(synchronousFailure.preflight.check(synchronousFailure.context))
                .thenThrow(new IllegalStateException("synchronous preflight failure"));
        StepVerifier.create(synchronousFailure.invoker.call(
                        synchronousFailure.spec, List.of(), synchronousFailure.context))
                .expectErrorMessage("synchronous preflight failure")
                .verify();
        verify(synchronousFailure.agent, never())
                .call(any(List.class), any(RuntimeContext.class));
        verify(synchronousFailure.lease, timeout(1000)).close();
    }

    @Test
    void queuedSameSlotCallDoesNotAcquireAHarnessLeaseOrRunPreflight() {
        InvocationFixture fixture = fixture();
        when(fixture.agent.call(any(List.class), any(RuntimeContext.class)))
                .thenReturn(Mono.never());
        clearInvocations(fixture.cache, fixture.preflight, fixture.lease, fixture.agent);

        Disposable active = fixture.invoker.call(
                        fixture.spec, List.of(), fixture.context)
                .subscribe();
        verify(fixture.cache, times(1)).acquire(fixture.spec);
        verify(fixture.preflight, times(1)).check(fixture.context);

        Disposable queued = fixture.invoker.call(
                        fixture.spec, List.of(), fixture.context)
                .subscribe();
        verify(fixture.cache, times(1)).acquire(fixture.spec);
        verify(fixture.preflight, times(1)).check(fixture.context);

        queued.dispose();
        active.dispose();
        verify(fixture.lease, timeout(1000).times(1)).close();
    }

    @Test
    void cleanupRunsOnTheModelBlockingScheduler() {
        InvocationFixture fixture = fixture();
        Msg response = mock(Msg.class);
        AtomicReference<String> cleanupThread = new AtomicReference<>();
        when(fixture.agent.call(any(List.class), any(RuntimeContext.class)))
                .thenReturn(Mono.just(response));
        org.mockito.Mockito.doAnswer(ignored -> {
            cleanupThread.set(Thread.currentThread().getName());
            return null;
        }).when(fixture.lease).close();

        StepVerifier.create(fixture.invoker.call(fixture.spec, List.of(), fixture.context))
                .expectNext(response)
                .verifyComplete();

        assertThat(cleanupThread.get()).startsWith("invoker-model-test-");
    }

    @Test
    void cleanupFallsBackToDirectReleaseWhenTheModelSchedulerRejects() {
        Scheduler rejectedScheduler = Schedulers.newSingle("rejected-cleanup-test");
        rejectedScheduler.dispose();
        InvocationFixture fixture = fixture(rejectedScheduler);
        Msg response = mock(Msg.class);
        when(fixture.agent.call(any(List.class), any(RuntimeContext.class)))
                .thenReturn(Mono.just(response));

        StepVerifier.create(fixture.invoker.call(fixture.spec, List.of(), fixture.context))
                .expectNext(response)
                .verifyComplete();

        verify(fixture.lease, times(1)).close();
    }

    private InvocationFixture fixture() {
        return fixture(modelScheduler);
    }

    private InvocationFixture fixture(Scheduler cleanupScheduler) {
        HarnessLeaseCache cache = mock(HarnessLeaseCache.class);
        AgentStatePreflight preflight = mock(AgentStatePreflight.class);
        HarnessLease lease = mock(HarnessLease.class);
        AgentKernelResource resource = mock(AgentKernelResource.class);
        HarnessAgent agent = mock(HarnessAgent.class, Answers.RETURNS_DEEP_STUBS);
        AgentKernelSpec spec = mock(AgentKernelSpec.class);
        RuntimeContext context = mock(RuntimeContext.class);
        when(context.getUserId()).thenReturn("42");
        when(context.getSessionId()).thenReturn("conversation-7");
        when(context.get(ToolPermissionContext.class))
                .thenReturn(new ToolPermissionContext(ToolExecutionMode.DEFAULT));
        when(cache.acquire(spec)).thenReturn(Mono.just(lease));
        when(preflight.check(context)).thenReturn(Mono.empty());
        when(lease.resource()).thenReturn(resource);
        when(resource.agent()).thenReturn(agent);
        return new InvocationFixture(
                new DefaultAgentScopeHarnessInvoker(cache, preflight, cleanupScheduler),
                cache, preflight, lease, resource, agent, spec, context);
    }

    private record InvocationFixture(
            DefaultAgentScopeHarnessInvoker invoker,
            HarnessLeaseCache cache,
            AgentStatePreflight preflight,
            HarnessLease lease,
            AgentKernelResource resource,
            HarnessAgent agent,
            AgentKernelSpec spec,
            RuntimeContext context) {
    }
}
