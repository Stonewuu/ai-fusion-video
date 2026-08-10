package com.stonewu.fusion.service.ai.agentscope.runtime;

import com.stonewu.fusion.config.AgentScopeRuntimeProperties;
import com.stonewu.fusion.service.ai.run.AgentRuntimeMetrics;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Owns the four bounded schedulers used at blocking AgentScope boundaries.
 */
public final class AgentRuntimeSchedulers implements AutoCloseable {

    public static final int STATE_QUEUE_CAPACITY = 512;
    public static final int JOURNAL_QUEUE_CAPACITY = 2048;
    public static final int MODEL_QUEUE_CAPACITY = 256;
    public static final int TOOL_QUEUE_CAPACITY = 256;

    public static final String STATE_OVERLOAD_CODE = "AGENT_STATE_SCHEDULER_OVERLOADED";
    public static final String JOURNAL_OVERLOAD_CODE = "AGENT_JOURNAL_SCHEDULER_OVERLOADED";
    public static final String MODEL_OVERLOAD_CODE = "AGENT_MODEL_BLOCKING_SCHEDULER_OVERLOADED";
    public static final String TOOL_OVERLOAD_CODE = "AGENT_TOOL_BLOCKING_SCHEDULER_OVERLOADED";

    private final OwnedScheduler state;
    private final OwnedScheduler journal;
    private final OwnedScheduler modelBlocking;
    private final OwnedScheduler toolBlocking;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final AgentRuntimeMetrics metrics;

    public AgentRuntimeSchedulers(AgentScopeRuntimeProperties properties) {
        this(properties, AgentRuntimeMetrics.noop());
    }

    public AgentRuntimeSchedulers(
            AgentScopeRuntimeProperties properties,
            AgentRuntimeMetrics metrics) {
        Objects.requireNonNull(properties, "properties must not be null");
        this.metrics = Objects.requireNonNull(metrics, "metrics must not be null");
        state = fixedScheduler(
                "agent-state", properties.getStateThreads(), STATE_QUEUE_CAPACITY, STATE_OVERLOAD_CODE);
        journal = fixedScheduler(
                "agent-journal", properties.getJournalThreads(), JOURNAL_QUEUE_CAPACITY, JOURNAL_OVERLOAD_CODE);
        modelBlocking = fixedScheduler(
                "agent-model-blocking", properties.getModelThreads(), MODEL_QUEUE_CAPACITY, MODEL_OVERLOAD_CODE);
        toolBlocking = fixedScheduler(
                "agent-tool-blocking", properties.getToolThreads(), TOOL_QUEUE_CAPACITY, TOOL_OVERLOAD_CODE);
    }

    public Scheduler state() {
        return state.scheduler();
    }

    public Scheduler journal() {
        return journal.scheduler();
    }

    public Scheduler modelBlocking() {
        return modelBlocking.scheduler();
    }

    public Scheduler toolBlocking() {
        return toolBlocking.scheduler();
    }

    public boolean isClosed() {
        return closed.get();
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        close(state);
        close(journal);
        close(modelBlocking);
        close(toolBlocking);
    }

    private OwnedScheduler fixedScheduler(
            String name, int threads, int queueCapacity, String overloadCode) {
        ThreadPoolExecutor executor = new ThreadPoolExecutor(
                threads,
                threads,
                0L,
                TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(queueCapacity),
                namedThreadFactory(name),
                (task, rejectedExecutor) -> {
                    if (TOOL_OVERLOAD_CODE.equals(overloadCode)) {
                        metrics.toolSchedulerViolation();
                    } else if (STATE_OVERLOAD_CODE.equals(overloadCode)) {
                        metrics.stateBulkheadRejected();
                    }
                    throw new SchedulerOverloadedException(overloadCode, name, rejectedExecutor.isShutdown());
                });
        executor.prestartAllCoreThreads();
        return new OwnedScheduler(executor, Schedulers.fromExecutorService(executor, name));
    }

    private ThreadFactory namedThreadFactory(String prefix) {
        AtomicInteger sequence = new AtomicInteger();
        return task -> {
            Thread thread = new Thread(task, prefix + "-" + sequence.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        };
    }

    private void close(OwnedScheduler owned) {
        owned.scheduler().dispose();
        owned.executor().shutdownNow();
    }

    private record OwnedScheduler(ThreadPoolExecutor executor, Scheduler scheduler) {
    }

    public static final class SchedulerOverloadedException extends RejectedExecutionException {

        private final String code;
        private final String schedulerName;

        public SchedulerOverloadedException(String code, String schedulerName, boolean shutdown) {
            super(code + ": scheduler=" + schedulerName + ", shutdown=" + shutdown);
            this.code = code;
            this.schedulerName = schedulerName;
        }

        public String getCode() {
            return code;
        }

        public String getSchedulerName() {
            return schedulerName;
        }
    }
}
