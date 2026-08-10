package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.mockito.stubbing.Answer;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;
import reactor.test.StepVerifier;

import java.time.Duration;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class HarnessLeaseCacheTests {
    private final Scheduler scheduler = Schedulers.newParallel("lease-cache-test", 4);

    @AfterEach
    void closeScheduler() {
        scheduler.dispose();
    }

    @Test
    void createsSameKeyOnceAndReleasesEveryLeaseIdempotently() {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource resource = mock(AgentKernelResource.class);
        when(factory.create(any())).thenReturn(resource);
        HarnessLeaseCache cache = cache(factory, 4, Duration.ofMillis(100));
        AgentKernelSpec spec = spec("agent-a", "model-a");

        StepVerifier.create(Flux.range(0, 20)
                        .flatMap(ignored -> cache.acquire(spec), 20)
                        .collectList())
                .assertNext(leases -> {
                    assertThat(leases).hasSize(20);
                    assertThat(leases).allSatisfy(lease ->
                            assertThat(lease.resource()).isSameAs(resource));
                    leases.forEach(lease -> {
                        lease.close();
                        lease.close();
                    });
                })
                .verifyComplete();

        verify(factory, times(1)).create(any());
        assertThat(cache.activeLeaseCount()).isZero();
    }

    @Test
    void hardCapacityWaitsThenReturns503WithoutClosingActiveEntries() {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource first = mock(AgentKernelResource.class);
        AgentKernelResource second = mock(AgentKernelResource.class);
        when(factory.create(any())).thenReturn(first, second);
        HarnessLeaseCache cache = cache(factory, 2, Duration.ofMillis(60));
        HarnessLease firstLease = cache.acquire(spec("agent-a", "model-a")).block();
        HarnessLease secondLease = cache.acquire(spec("agent-b", "model-b")).block();

        long started = System.nanoTime();
        StepVerifier.create(cache.acquire(spec("agent-c", "model-c")))
                .expectErrorSatisfies(error -> {
                    assertThat(error).isInstanceOf(BusinessException.class);
                    assertThat(((BusinessException) error).getCode()).isEqualTo(503);
                    assertThat(error).hasMessageContaining("HARNESS_CAPACITY_EXHAUSTED");
                })
                .verify();

        assertThat(Duration.ofNanos(System.nanoTime() - started)).isGreaterThanOrEqualTo(Duration.ofMillis(50));
        verify(first, never()).close();
        verify(second, never()).close();
        firstLease.close();
        secondLease.close();
    }

    @Test
    void evictsLeastRecentlyUsedIdleEntryButNeverAnActiveEntry() {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource first = mock(AgentKernelResource.class);
        AgentKernelResource second = mock(AgentKernelResource.class);
        when(factory.create(any())).thenReturn(first, second);
        HarnessLeaseCache cache = cache(factory, 1, Duration.ofMillis(100));
        HarnessLease firstLease = cache.acquire(spec("agent-a", "model-a")).block();
        firstLease.close();

        HarnessLease secondLease = cache.acquire(spec("agent-b", "model-b")).block();

        verify(first, times(1)).close();
        verify(second, never()).close();
        secondLease.close();
    }

    @Test
    void acquireIsLazyAndDrainWaitsForActiveLeaseThenClosesOnce() {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource resource = mock(AgentKernelResource.class);
        when(factory.create(any())).thenReturn(resource);
        HarnessLeaseCache cache = cache(factory, 1, Duration.ofMillis(100));
        cache.acquire(spec("never-subscribed", "model-x"));
        verify(factory, never()).create(any());
        HarnessLease lease = cache.acquire(spec("agent-a", "model-a")).block();

        StepVerifier.create(cache.drainAndClose(Duration.ofSeconds(1)))
                .then(lease::close)
                .verifyComplete();

        verify(resource, times(1)).close();
        StepVerifier.create(cache.acquire(spec("agent-b", "model-b")))
                .expectErrorSatisfies(error ->
                        assertThat(error).hasMessageContaining("HARNESS_SHUTTING_DOWN"))
                .verify();
        StepVerifier.create(cache.drainAndClose(Duration.ofSeconds(1))).verifyComplete();
        verify(resource, times(1)).close();
    }

    @Test
    void drainTimeoutDoesNotCloseActiveResource() {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource resource = mock(AgentKernelResource.class);
        when(factory.create(any())).thenReturn(resource);
        HarnessLeaseCache cache = cache(factory, 1, Duration.ofMillis(100));
        HarnessLease lease = cache.acquire(spec("agent-a", "model-a")).block();

        StepVerifier.create(cache.drainAndClose(Duration.ofMillis(40)))
                .expectErrorSatisfies(error ->
                        assertThat(error).hasMessageContaining("HARNESS_DRAIN_TIMEOUT"))
                .verify();
        verify(resource, never()).close();

        lease.close();
        verify(resource, times(1)).close();
    }

    @Test
    void factoryFailureRestoresPermitAndAllowsSameKeyRetry() {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource resource = mock(AgentKernelResource.class);
        when(factory.create(any()))
                .thenThrow(new IllegalStateException("factory failed"))
                .thenReturn(resource);
        HarnessLeaseCache cache = cache(factory, 1, Duration.ofMillis(100));
        AgentKernelSpec spec = spec("agent-a", "model-a");

        StepVerifier.create(cache.acquire(spec))
                .expectErrorMessage("factory failed")
                .verify();

        HarnessLease lease = cache.acquire(spec).block();
        assertThat(lease).isNotNull();
        assertThat(cache.size()).isEqualTo(1);
        verify(factory, times(2)).create(any());
        lease.close();
    }

    @Test
    void closeFailureStillRestoresPermitForAReplacementEntry() {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource first = mock(AgentKernelResource.class);
        AgentKernelResource second = mock(AgentKernelResource.class);
        when(factory.create(any())).thenReturn(first, second);
        org.mockito.Mockito.doThrow(new IllegalStateException("close failed"))
                .when(first).close();
        HarnessLeaseCache cache = cache(factory, 1, Duration.ofMillis(100));
        HarnessLease firstLease = cache.acquire(spec("agent-a", "model-a")).block();
        firstLease.close();

        StepVerifier.create(cache.acquire(spec("agent-b", "model-b")))
                .expectErrorMessage("close failed")
                .verify();

        HarnessLease secondLease = cache.acquire(spec("agent-b", "model-b")).block();
        assertThat(secondLease).isNotNull();
        verify(factory, times(2)).create(any());
        secondLease.close();
    }

    @Test
    void expiresAnIdleSameKeyEntryBeforeReusingIt() {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource first = mock(AgentKernelResource.class);
        AgentKernelResource second = mock(AgentKernelResource.class);
        when(factory.create(any())).thenReturn(first, second);
        AtomicLong ticker = new AtomicLong(1L);
        HarnessLeaseCache cache = new HarnessLeaseCache(
                factory,
                scheduler,
                1,
                Duration.ofNanos(10),
                Duration.ofMillis(100),
                Duration.ofMillis(5),
                ticker::get);
        AgentKernelSpec spec = spec("agent-a", "model-a");
        HarnessLease firstLease = cache.acquire(spec).block();
        firstLease.close();
        ticker.addAndGet(10L);

        HarnessLease secondLease = cache.acquire(spec).block();

        assertThat(secondLease.resource()).isSameAs(second);
        verify(first, times(1)).close();
        verify(factory, times(2)).create(any());
        secondLease.close();
    }

    @Test
    void cancellationWhileFactoryReturnsDoesNotLeakTheLease() throws Exception {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource resource = mock(AgentKernelResource.class);
        CountDownLatch factoryStarted = new CountDownLatch(1);
        CountDownLatch allowFactoryReturn = new CountDownLatch(1);
        when(factory.create(any())).thenAnswer(ignored -> {
            factoryStarted.countDown();
            try {
                allowFactoryReturn.await(2, TimeUnit.SECONDS);
            } catch (InterruptedException cancellation) {
                Thread.interrupted();
            }
            return resource;
        });
        HarnessLeaseCache cache = cache(factory, 1, Duration.ofMillis(100));

        Disposable subscription = cache.acquire(spec("agent-a", "model-a")).subscribe();
        assertThat(factoryStarted.await(2, TimeUnit.SECONDS)).isTrue();
        subscription.dispose();
        allowFactoryReturn.countDown();

        StepVerifier.create(cache.drainAndClose(Duration.ofSeconds(1)))
                .verifyComplete();
        assertThat(cache.activeLeaseCount()).isZero();
        verify(resource, times(1)).close();
    }

    @Test
    void drainImmediatelyRejectsNewAcquiresAndWaitsForAnInFlightFactoryLease() throws Exception {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource resource = mock(AgentKernelResource.class);
        CountDownLatch factoryStarted = new CountDownLatch(1);
        CountDownLatch allowFactoryReturn = new CountDownLatch(1);
        when(factory.create(any())).thenAnswer(ignored -> {
            factoryStarted.countDown();
            assertThat(allowFactoryReturn.await(2, TimeUnit.SECONDS)).isTrue();
            return resource;
        });
        HarnessLeaseCache cache = cache(factory, 2, Duration.ofMillis(100));
        CompletableFuture<HarnessLease> inFlightAcquire =
                cache.acquire(spec("agent-a", "model-a")).toFuture();
        assertThat(factoryStarted.await(2, TimeUnit.SECONDS)).isTrue();

        CompletableFuture<Void> drain = cache.drainAndClose(Duration.ofSeconds(1)).toFuture();
        StepVerifier.create(cache.acquire(spec("agent-b", "model-b")))
                .expectErrorSatisfies(error ->
                        assertThat(error).hasMessageContaining("HARNESS_SHUTTING_DOWN"))
                .verify();
        allowFactoryReturn.countDown();
        HarnessLease lease = inFlightAcquire.get(2, TimeUnit.SECONDS);
        assertThat(drain).isNotDone();

        lease.close();
        drain.get(2, TimeUnit.SECONDS);
        verify(resource, times(1)).close();
    }

    @Test
    void drainReportsCloseFailureFromTheFinalLeaseRelease() throws Exception {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource resource = mock(AgentKernelResource.class);
        when(factory.create(any())).thenReturn(resource);
        org.mockito.Mockito.doThrow(new IllegalStateException("drain close failed"))
                .when(resource).close();
        HarnessLeaseCache cache = cache(factory, 1, Duration.ofMillis(100));
        HarnessLease lease = cache.acquire(spec("agent-a", "model-a")).block();
        CompletableFuture<Void> drain = cache.drainAndClose(Duration.ofSeconds(1)).toFuture();

        assertThatThrownBy(lease::close)
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("drain close failed");
        assertThatThrownBy(() -> drain.get(2, TimeUnit.SECONDS))
                .hasRootCauseMessage("drain close failed");
        verify(resource, times(1)).close();
    }

    @Test
    void drainWaitsForEveryConcurrentReleaseToFinishClosing() throws Exception {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource slowFailure = mock(AgentKernelResource.class);
        AgentKernelResource fast = mock(AgentKernelResource.class);
        when(factory.create(any())).thenReturn(slowFailure, fast);
        CountDownLatch slowCloseStarted = new CountDownLatch(1);
        CountDownLatch allowSlowClose = new CountDownLatch(1);
        org.mockito.Mockito.doAnswer(ignored -> {
            slowCloseStarted.countDown();
            assertThat(allowSlowClose.await(2, TimeUnit.SECONDS)).isTrue();
            throw new IllegalStateException("slow close failed");
        }).when(slowFailure).close();
        HarnessLeaseCache cache = cache(factory, 2, Duration.ofMillis(100));
        HarnessLease slowLease = cache.acquire(spec("agent-a", "model-a")).block();
        HarnessLease fastLease = cache.acquire(spec("agent-b", "model-b")).block();
        CompletableFuture<Void> drain = cache.drainAndClose(Duration.ofSeconds(1)).toFuture();

        CompletableFuture<Void> slowRelease = CompletableFuture.runAsync(() ->
                assertThatThrownBy(slowLease::close).hasMessage("slow close failed"));
        assertThat(slowCloseStarted.await(2, TimeUnit.SECONDS)).isTrue();
        fastLease.close();
        assertThat(drain).isNotDone();

        allowSlowClose.countDown();
        slowRelease.get(2, TimeUnit.SECONDS);
        assertThatThrownBy(() -> drain.get(2, TimeUnit.SECONDS))
                .hasRootCauseMessage("slow close failed");
    }

    @Test
    void drainTimeoutClosesIdleEntriesButLeavesActiveEntryUntilRelease() {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource idle = mock(AgentKernelResource.class);
        AgentKernelResource active = mock(AgentKernelResource.class);
        when(factory.create(any())).thenReturn(idle, active);
        HarnessLeaseCache cache = cache(factory, 2, Duration.ofMillis(100));
        HarnessLease idleLease = cache.acquire(spec("agent-a", "model-a")).block();
        idleLease.close();
        HarnessLease activeLease = cache.acquire(spec("agent-b", "model-b")).block();

        StepVerifier.create(cache.drainAndClose(Duration.ofMillis(40)))
                .expectErrorSatisfies(error ->
                        assertThat(error).hasMessageContaining("HARNESS_DRAIN_TIMEOUT"))
                .verify();

        verify(idle, timeout(1000).times(1)).close();
        verify(active, never()).close();
        activeLease.close();
        verify(active, times(1)).close();
    }

    @Test
    void drainTimeoutAlsoBoundsAStuckResourceClose() throws Exception {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource first = mock(AgentKernelResource.class);
        AgentKernelResource second = mock(AgentKernelResource.class);
        when(factory.create(any())).thenReturn(first, second);
        CountDownLatch closeStarted = new CountDownLatch(1);
        CountDownLatch otherCloseCompleted = new CountDownLatch(1);
        CountDownLatch allowClose = new CountDownLatch(1);
        AtomicBoolean firstClose = new AtomicBoolean(true);
        Answer<Void> closeAnswer = ignored -> {
            if (firstClose.compareAndSet(true, false)) {
                closeStarted.countDown();
                boolean released = false;
                while (!released) {
                    try {
                        released = allowClose.await(2, TimeUnit.SECONDS);
                    } catch (InterruptedException cancellation) {
                        Thread.interrupted();
                    }
                }
            } else {
                otherCloseCompleted.countDown();
            }
            return null;
        };
        org.mockito.Mockito.doAnswer(closeAnswer).when(first).close();
        org.mockito.Mockito.doAnswer(closeAnswer).when(second).close();
        HarnessLeaseCache cache = cache(factory, 2, Duration.ofMillis(100));
        HarnessLease firstLease = cache.acquire(spec("agent-a", "model-a")).block();
        HarnessLease secondLease = cache.acquire(spec("agent-b", "model-b")).block();
        firstLease.close();
        secondLease.close();

        CompletableFuture<Void> drain = cache.drainAndClose(Duration.ofMillis(40)).toFuture();
        assertThat(closeStarted.await(2, TimeUnit.SECONDS)).isTrue();
        assertThatThrownBy(() -> drain.get(1, TimeUnit.SECONDS))
                .hasRootCauseMessage("HARNESS_DRAIN_TIMEOUT");
        assertThat(otherCloseCompleted.await(1, TimeUnit.SECONDS)).isTrue();

        allowClose.countDown();
        verify(first, timeout(1000).times(1)).close();
        verify(second, timeout(1000).times(1)).close();
    }

    @Test
    void timeoutStillTerminatesWhenModelSchedulerRejectsAStuckCleanup() throws Exception {
        AgentScopeHarnessFactory factory = mock(AgentScopeHarnessFactory.class);
        AgentKernelResource idle = mock(AgentKernelResource.class);
        AgentKernelResource active = mock(AgentKernelResource.class);
        when(factory.create(any())).thenReturn(idle, active);
        CountDownLatch closeStarted = new CountDownLatch(1);
        CountDownLatch allowClose = new CountDownLatch(1);
        org.mockito.Mockito.doAnswer(ignored -> {
            closeStarted.countDown();
            boolean released = false;
            while (!released) {
                try {
                    released = allowClose.await(2, TimeUnit.SECONDS);
                } catch (InterruptedException cancellation) {
                    Thread.interrupted();
                }
            }
            return null;
        }).when(idle).close();
        RejectingScheduler rejectingScheduler = new RejectingScheduler(scheduler);
        HarnessLeaseCache cache = new HarnessLeaseCache(
                factory, rejectingScheduler, 2, Duration.ofMillis(100), Duration.ofMillis(5));
        HarnessLease idleLease = cache.acquire(spec("agent-a", "model-a")).block();
        idleLease.close();
        HarnessLease activeLease = cache.acquire(spec("agent-b", "model-b")).block();
        CompletableFuture<Void> drain = cache.drainAndClose(Duration.ofMillis(40)).toFuture();
        rejectingScheduler.rejectNewTasks();

        try {
            assertThat(closeStarted.await(1, TimeUnit.SECONDS)).isTrue();
            assertThatThrownBy(() -> drain.get(500, TimeUnit.MILLISECONDS))
                    .hasRootCauseMessage("HARNESS_DRAIN_TIMEOUT");
        } finally {
            allowClose.countDown();
            activeLease.close();
        }
    }

    private HarnessLeaseCache cache(
            AgentScopeHarnessFactory factory, int maximumSize, Duration wait) {
        return new HarnessLeaseCache(factory, scheduler, maximumSize, wait, Duration.ofMillis(5));
    }

    private static AgentKernelSpec spec(String definition, String modelFingerprint) {
        AiModel model = AiModel.builder().id((long) definition.hashCode()).code(modelFingerprint).build();
        String prompt = "prompt for " + definition;
        AgentKernelKey key = AgentKernelKey.create(
                definition, modelFingerprint, AgentKernelKey.promptVersion(prompt),
                List.of(), "afv-tools-v1");
        return new AgentKernelSpec(
                key, model, definition, definition, "description", prompt, 5,
                List.of(), Set.of(), "afv-tools-v1");
    }

    private static final class RejectingScheduler implements Scheduler {
        private final Scheduler delegate;
        private final AtomicBoolean rejecting = new AtomicBoolean();

        private RejectingScheduler(Scheduler delegate) {
            this.delegate = delegate;
        }

        private void rejectNewTasks() {
            rejecting.set(true);
        }

        @Override
        public Disposable schedule(Runnable task) {
            if (rejecting.get()) {
                throw new RejectedExecutionException("test scheduler rejected task");
            }
            return delegate.schedule(task);
        }

        @Override
        public Worker createWorker() {
            Worker worker = delegate.createWorker();
            return new Worker() {
                @Override
                public Disposable schedule(Runnable task) {
                    if (rejecting.get()) {
                        throw new RejectedExecutionException("test worker rejected task");
                    }
                    return worker.schedule(task);
                }

                @Override
                public void dispose() {
                    worker.dispose();
                }

                @Override
                public boolean isDisposed() {
                    return worker.isDisposed();
                }
            };
        }

        @Override
        public void dispose() {
            delegate.dispose();
        }

        @Override
        public boolean isDisposed() {
            return delegate.isDisposed();
        }
    }
}
