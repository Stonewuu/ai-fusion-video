package com.stonewu.fusion.service.ai.run;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.ObjectProvider;
import reactor.core.publisher.Sinks;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentScopeKernelLifecycleTests {

    @Test
    void usesShutdownPortWhenAvailableAndAlwaysInvokesCallbackAsynchronously() {
        AgentRuntimeShutdownPort port = mock(AgentRuntimeShutdownPort.class);
        ObjectProvider<AgentRuntimeShutdownPort> provider = provider(port);
        Sinks.Empty<Void> completion = Sinks.empty();
        when(port.shutdown(Duration.ofSeconds(2))).thenReturn(completion.asMono());
        AgentScopeKernelLifecycle lifecycle =
                new AgentScopeKernelLifecycle(provider, Duration.ofSeconds(2));
        lifecycle.start();
        AtomicInteger callbacks = new AtomicInteger();

        lifecycle.stop(callbacks::incrementAndGet);
        assertThat(callbacks).hasValue(0);
        completion.tryEmitEmpty();

        assertThat(callbacks).hasValue(1);
    }

    @Test
    void missingShutdownPortFailsClosedAndInvokesCallback() {
        AgentScopeKernelLifecycle lifecycle = new AgentScopeKernelLifecycle(
                provider(null), Duration.ofSeconds(1));
        lifecycle.start();
        AtomicInteger callbacks = new AtomicInteger();

        lifecycle.stop(callbacks::incrementAndGet);

        assertThat(callbacks).hasValue(1);
        assertThat(lifecycle.isRunning()).isFalse();
    }

    @Test
    void concurrentStopsShareOneShutdownAndEachCallbackWaitsForItsTerminalSignal() {
        AgentRuntimeShutdownPort port = mock(AgentRuntimeShutdownPort.class);
        Sinks.Empty<Void> completion = Sinks.empty();
        when(port.shutdown(Duration.ofSeconds(2))).thenReturn(completion.asMono());
        AgentScopeKernelLifecycle lifecycle = new AgentScopeKernelLifecycle(
                provider(port), Duration.ofSeconds(2));
        lifecycle.start();
        AtomicInteger firstCallbacks = new AtomicInteger();
        AtomicInteger secondCallbacks = new AtomicInteger();

        lifecycle.stop(firstCallbacks::incrementAndGet);
        lifecycle.stop(secondCallbacks::incrementAndGet);

        assertThat(firstCallbacks).hasValue(0);
        assertThat(secondCallbacks).hasValue(0);
        verify(port, times(1)).shutdown(Duration.ofSeconds(2));
        completion.tryEmitEmpty();
        assertThat(firstCallbacks).hasValue(1);
        assertThat(secondCallbacks).hasValue(1);
    }

    @Test
    void synchronousShutdownFailureStillCompletesEveryCallbackExactlyOnce() {
        AgentRuntimeShutdownPort port = mock(AgentRuntimeShutdownPort.class);
        when(port.shutdown(Duration.ofSeconds(1)))
                .thenThrow(new IllegalStateException("sync shutdown failure"));
        AgentScopeKernelLifecycle lifecycle = new AgentScopeKernelLifecycle(
                provider(port), Duration.ofSeconds(1));
        lifecycle.start();
        AtomicInteger callbacks = new AtomicInteger();

        lifecycle.stop(callbacks::incrementAndGet);
        lifecycle.stop(callbacks::incrementAndGet);

        assertThat(callbacks).hasValue(2);
        verify(port, times(1)).shutdown(Duration.ofSeconds(1));
    }

    @SuppressWarnings("unchecked")
    private static ObjectProvider<AgentRuntimeShutdownPort> provider(AgentRuntimeShutdownPort port) {
        ObjectProvider<AgentRuntimeShutdownPort> provider = mock(ObjectProvider.class);
        when(provider.getIfAvailable()).thenReturn(port);
        return provider;
    }
}
