package com.stonewu.fusion.service.ai.agentscope.runtime;

import com.stonewu.fusion.config.AgentScopeRuntimeProperties;
import org.junit.jupiter.api.Test;
import reactor.core.scheduler.Scheduler;
import reactor.test.StepVerifier;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class AgentRuntimeSchedulersTests {

    @Test
    void computesExactThreadDefaultsFromProcessorCount() {
        assertThat(AgentScopeRuntimeProperties.defaultStateThreads(1)).isEqualTo(4);
        assertThat(AgentScopeRuntimeProperties.defaultJournalThreads(8)).isEqualTo(16);
        assertThat(AgentScopeRuntimeProperties.defaultStateThreads(64)).isEqualTo(32);
        assertThat(AgentScopeRuntimeProperties.defaultModelThreads(1)).isEqualTo(8);
        assertThat(AgentScopeRuntimeProperties.defaultToolThreads(8)).isEqualTo(32);
        assertThat(AgentScopeRuntimeProperties.defaultModelThreads(64)).isEqualTo(64);
        assertThat(AgentScopeRuntimeProperties.defaultStateThreads(Integer.MAX_VALUE)).isEqualTo(32);
        assertThat(AgentScopeRuntimeProperties.defaultToolThreads(Integer.MAX_VALUE)).isEqualTo(64);
    }

    @Test
    void rejectsNonPositiveThreadOverrides() {
        AgentScopeRuntimeProperties properties = new AgentScopeRuntimeProperties();

        assertThatThrownBy(() -> properties.setStateThreads(0))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> properties.setJournalThreads(-1))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> properties.setModelThreads(0))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> properties.setToolThreads(-1))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void executesOnAllFourOwnedThreadPrefixes() {
        try (AgentRuntimeSchedulers schedulers = singleThreadSchedulers()) {
            assertThreadPrefix(schedulers.state(), "agent-state-");
            assertThreadPrefix(schedulers.journal(), "agent-journal-");
            assertThreadPrefix(schedulers.modelBlocking(), "agent-model-blocking-");
            assertThreadPrefix(schedulers.toolBlocking(), "agent-tool-blocking-");
        }
    }

    @Test
    void eachSchedulerUsesItsExactBoundedQueueAndSpecificOverloadCode() throws Exception {
        try (AgentRuntimeSchedulers schedulers = singleThreadSchedulers()) {
            assertBoundedRejection(
                    schedulers.state(),
                    AgentRuntimeSchedulers.STATE_QUEUE_CAPACITY,
                    AgentRuntimeSchedulers.STATE_OVERLOAD_CODE);
            assertBoundedRejection(
                    schedulers.journal(),
                    AgentRuntimeSchedulers.JOURNAL_QUEUE_CAPACITY,
                    AgentRuntimeSchedulers.JOURNAL_OVERLOAD_CODE);
            assertBoundedRejection(
                    schedulers.modelBlocking(),
                    AgentRuntimeSchedulers.MODEL_QUEUE_CAPACITY,
                    AgentRuntimeSchedulers.MODEL_OVERLOAD_CODE);
            assertBoundedRejection(
                    schedulers.toolBlocking(),
                    AgentRuntimeSchedulers.TOOL_QUEUE_CAPACITY,
                    AgentRuntimeSchedulers.TOOL_OVERLOAD_CODE);
        }
    }

    @Test
    void closeIsIdempotentAndClosedSchedulersReject() {
        AgentRuntimeSchedulers schedulers = singleThreadSchedulers();

        schedulers.close();
        schedulers.close();

        assertThat(schedulers.isClosed()).isTrue();
        for (Scheduler scheduler : List.of(
                schedulers.state(), schedulers.journal(),
                schedulers.modelBlocking(), schedulers.toolBlocking())) {
            assertThat(scheduler.isDisposed()).isTrue();
            assertThatThrownBy(() -> scheduler.schedule(() -> {
            })).isInstanceOf(RuntimeException.class);
        }
    }

    private AgentRuntimeSchedulers singleThreadSchedulers() {
        AgentScopeRuntimeProperties properties = new AgentScopeRuntimeProperties();
        properties.setStateThreads(1);
        properties.setJournalThreads(1);
        properties.setModelThreads(1);
        properties.setToolThreads(1);
        return new AgentRuntimeSchedulers(properties);
    }

    private void assertThreadPrefix(Scheduler scheduler, String prefix) {
        StepVerifier.create(reactor.core.publisher.Mono.fromCallable(
                        () -> Thread.currentThread().getName())
                .subscribeOn(scheduler))
                .assertNext(name -> assertThat(name).startsWith(prefix))
                .verifyComplete();
    }

    private void assertBoundedRejection(
            Scheduler scheduler, int queueCapacity, String expectedCode) throws Exception {
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch callerRan = new CountDownLatch(1);
        Scheduler.Worker worker = scheduler.createWorker();
        try {
            worker.schedule(() -> {
                started.countDown();
                try {
                    release.await();
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                }
            });
            assertThat(started.await(2, TimeUnit.SECONDS)).isTrue();
            for (int index = 0; index < queueCapacity; index++) {
                worker.schedule(() -> {
                });
            }

            assertThatThrownBy(() -> worker.schedule(callerRan::countDown))
                    .isInstanceOf(AgentRuntimeSchedulers.SchedulerOverloadedException.class)
                    .extracting(failure -> ((AgentRuntimeSchedulers.SchedulerOverloadedException) failure).getCode())
                    .isEqualTo(expectedCode);
            assertThat(callerRan.await(50, TimeUnit.MILLISECONDS)).isFalse();
        } finally {
            release.countDown();
            worker.dispose();
        }
    }
}
