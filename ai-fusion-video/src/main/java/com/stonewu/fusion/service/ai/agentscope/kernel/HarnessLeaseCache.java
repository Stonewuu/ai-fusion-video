package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Scheduler;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.LongSupplier;

@Component
@Slf4j
public final class HarnessLeaseCache {
    public static final int DEFAULT_MAXIMUM_SIZE = 64;
    public static final Duration DEFAULT_EXPIRE_AFTER_ACCESS = Duration.ofMinutes(30);
    public static final Duration DEFAULT_CAPACITY_WAIT = Duration.ofSeconds(5);
    public static final Duration DEFAULT_POLL_INTERVAL = Duration.ofMillis(10);

    private final AgentScopeHarnessFactory factory;
    private final Scheduler modelScheduler;
    private final long expireAfterAccessNanos;
    private final long capacityWaitNanos;
    private final long pollIntervalNanos;
    private final LongSupplier nanoTime;
    private final Semaphore capacityPermits;
    private final ConcurrentHashMap<AgentKernelKey, Entry> entries = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<AgentKernelKey, KeyLock> creationLocks = new ConcurrentHashMap<>();
    private final Object evictionLock = new Object();
    private final ReentrantReadWriteLock lifecycleLock = new ReentrantReadWriteLock();
    private final AtomicBoolean draining = new AtomicBoolean();
    private final AtomicInteger activeLeases = new AtomicInteger();
    private final AtomicInteger emergencyCleanupSequence = new AtomicInteger();
    private final AtomicReference<Sinks.Empty<Void>> zeroLeaseWaiter = new AtomicReference<>();
    private final AtomicReference<Mono<Void>> drainSignal = new AtomicReference<>();
    private final Queue<Throwable> drainCloseFailures = new ConcurrentLinkedQueue<>();

    @Autowired
    public HarnessLeaseCache(
            AgentScopeHarnessFactory factory,
            AgentRuntimeSchedulers schedulers,
            @Value("${fusion.agentscope.v2.cache.maximum-size:64}") int maximumSize,
            @Value("${fusion.agentscope.v2.cache.expire-after-access:30m}") Duration expireAfterAccess,
            @Value("${fusion.agentscope.v2.cache.capacity-wait:5s}") Duration capacityWait) {
        this(factory,
                Objects.requireNonNull(schedulers, "schedulers must not be null").modelBlocking(),
                maximumSize,
                expireAfterAccess,
                capacityWait,
                DEFAULT_POLL_INTERVAL,
                System::nanoTime);
    }

    HarnessLeaseCache(
            AgentScopeHarnessFactory factory,
            Scheduler modelScheduler,
            int maximumSize,
            Duration capacityWait,
            Duration pollInterval) {
        this(factory, modelScheduler, maximumSize, DEFAULT_EXPIRE_AFTER_ACCESS,
                capacityWait, pollInterval, System::nanoTime);
    }

    HarnessLeaseCache(
            AgentScopeHarnessFactory factory,
            Scheduler modelScheduler,
            int maximumSize,
            Duration expireAfterAccess,
            Duration capacityWait,
            Duration pollInterval,
            LongSupplier nanoTime) {
        this.factory = Objects.requireNonNull(factory, "factory must not be null");
        this.modelScheduler = Objects.requireNonNull(modelScheduler, "modelScheduler must not be null");
        if (maximumSize <= 0) {
            throw new IllegalArgumentException("maximumSize must be greater than zero");
        }
        this.expireAfterAccessNanos = requirePositiveNanos(expireAfterAccess, "expireAfterAccess");
        this.capacityWaitNanos = requirePositiveNanos(capacityWait, "capacityWait");
        this.pollIntervalNanos = requirePositiveNanos(pollInterval, "pollInterval");
        this.nanoTime = Objects.requireNonNull(nanoTime, "nanoTime must not be null");
        this.capacityPermits = new Semaphore(maximumSize, true);
    }

    public Mono<HarnessLease> acquire(AgentKernelSpec spec) {
        Objects.requireNonNull(spec, "spec must not be null");
        return Mono.defer(() -> {
            if (draining.get()) {
                return Mono.error(shuttingDown());
            }
            return acquireUntil(spec, nowNanos());
        }).doOnDiscard(HarnessLease.class, this::releaseLeaseAsync);
    }

    public Mono<Void> drainAndClose(Duration timeout) {
        Duration safeTimeout = requirePositive(timeout, "timeout");
        Mono<Void> existing = drainSignal.get();
        if (existing != null) {
            return existing;
        }
        Mono<Void> candidate = Mono.<Void>defer(() -> {
                    draining.set(true);
                    BusinessException timeoutFailure = new BusinessException(
                            503, "HARNESS_DRAIN_TIMEOUT");
                    Mono<Void> awaitInFlightCreates = Mono.<Void>fromRunnable(this::awaitInFlightCreates)
                            .subscribeOn(modelScheduler);
                    return awaitInFlightCreates
                            .then(awaitZeroActiveLeases())
                            .then(Mono.defer(this::closeAllIdleEntries))
                            .timeout(safeTimeout, Mono.<Void>error(timeoutFailure))
                            .onErrorResume(failure -> failure == timeoutFailure,
                                    failure -> cleanupAfterDrainTimeout(timeoutFailure));
                })
                .cache();
        return drainSignal.compareAndSet(null, candidate) ? candidate : drainSignal.get();
    }

    int activeLeaseCount() {
        return activeLeases.get();
    }

    int size() {
        return entries.size();
    }

    private Mono<HarnessLease> acquireUntil(AgentKernelSpec spec, long startedNanos) {
        return tryAcquireOnModelScheduler(spec)
                .doOnDiscard(AcquireAttempt.class, this::releaseAttemptAsync)
                .flatMap(attempt -> {
                    if (attempt.lease() != null) {
                        return Mono.just(attempt.lease());
                    }
                    if (draining.get()) {
                        return Mono.error(shuttingDown());
                    }
                    long elapsed = nowNanos() - startedNanos;
                    long remaining = capacityWaitNanos - elapsed;
                    if (remaining <= 0L) {
                        return Mono.error(new BusinessException(
                                503, "HARNESS_CAPACITY_EXHAUSTED"));
                    }
                    Duration delay = Duration.ofNanos(Math.min(remaining, pollIntervalNanos));
                    return Mono.delay(delay).then(acquireUntil(spec, startedNanos));
                });
    }

    private Mono<AcquireAttempt> tryAcquireOnModelScheduler(AgentKernelSpec spec) {
        return Mono.create(sink -> {
            AtomicBoolean cancelled = new AtomicBoolean();
            AtomicReference<AcquireAttempt> pending = new AtomicReference<>();
            AtomicReference<Disposable> scheduled = new AtomicReference<>();
            sink.onCancel(() -> {
                cancelled.set(true);
                Disposable task = scheduled.get();
                if (task != null) {
                    task.dispose();
                }
                closePendingAttempt(pending);
            });
            try {
                Disposable task = modelScheduler.schedule(() -> {
                    AcquireAttempt attempt;
                    try {
                        attempt = new AcquireAttempt(tryAcquire(spec));
                    } catch (Throwable failure) {
                        if (!cancelled.get()) {
                            sink.error(failure);
                        }
                        return;
                    }
                    pending.set(attempt);
                    if (cancelled.get()) {
                        closePendingAttempt(pending);
                        return;
                    }
                    sink.success(attempt);
                    pending.compareAndSet(attempt, null);
                });
                scheduled.set(task);
                if (cancelled.get()) {
                    task.dispose();
                    closePendingAttempt(pending);
                }
            } catch (Throwable schedulingFailure) {
                sink.error(schedulingFailure);
            }
        });
    }

    private void closePendingAttempt(AtomicReference<AcquireAttempt> pending) {
        AcquireAttempt attempt = pending.getAndSet(null);
        if (attempt != null) {
            releaseAttemptAsync(attempt);
        }
    }

    private void releaseAttemptAsync(AcquireAttempt attempt) {
        scheduleRelease(attempt::close, "Harness acquire attempt");
    }

    private void releaseLeaseAsync(HarnessLease lease) {
        scheduleRelease(lease::close, "Harness lease");
    }

    private void scheduleRelease(Runnable release, String resourceName) {
        Runnable guardedRelease = () -> {
            try {
                release.run();
            } catch (Throwable failure) {
                log.error("{} cleanup failed", resourceName, failure);
            }
        };
        try {
            modelScheduler.schedule(guardedRelease);
        } catch (RuntimeException schedulingFailure) {
            startEmergencyCleanup(guardedRelease, resourceName, schedulingFailure);
        }
    }

    private void startEmergencyCleanup(
            Runnable cleanup, String resourceName, RuntimeException schedulingFailure) {
        log.warn("Model scheduler rejected {} cleanup; using emergency virtual thread: {}",
                resourceName, schedulingFailure.toString());
        try {
            Thread.ofVirtual()
                    .name("agent-kernel-emergency-cleanup-"
                            + emergencyCleanupSequence.incrementAndGet())
                    .start(cleanup);
        } catch (RuntimeException startFailure) {
            if (startFailure != schedulingFailure) {
                startFailure.addSuppressed(schedulingFailure);
            }
            log.error("Failed to start emergency {} cleanup", resourceName, startFailure);
        }
    }

    private HarnessLease tryAcquire(AgentKernelSpec spec) {
        lifecycleLock.readLock().lock();
        try {
            if (draining.get()) {
                throw shuttingDown();
            }
            HarnessLease existing = acquireExisting(spec.key());
            if (existing != null) {
                return existing;
            }

            KeyLock keyLock = retainKeyLock(spec.key());
            try {
                synchronized (keyLock) {
                    if (draining.get()) {
                        throw shuttingDown();
                    }
                    existing = acquireExisting(spec.key());
                    if (existing != null) {
                        return existing;
                    }
                    evictExpiredIdleEntries();
                    if (!capacityPermits.tryAcquire()) {
                        evictOneLeastRecentlyUsedIdleEntry();
                        if (!capacityPermits.tryAcquire()) {
                            return null;
                        }
                    }
                    Entry created = null;
                    try {
                        created = new Entry(spec.key(), factory.create(spec));
                        entries.put(spec.key(), created);
                        HarnessLease lease = created.tryAcquire();
                        if (lease == null) {
                            throw new IllegalStateException("New Harness cache entry rejected its first lease");
                        }
                        return lease;
                    } catch (Throwable failure) {
                        if (created == null) {
                            capacityPermits.release();
                        } else {
                            discardFailedCreation(created, failure);
                        }
                        throw failure;
                    }
                }
            } finally {
                releaseKeyLock(spec.key(), keyLock);
            }
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    private HarnessLease acquireExisting(AgentKernelKey key) {
        Entry entry = entries.get(key);
        if (entry != null && entry.isExpiredIdle(nowNanos(), expireAfterAccessNanos)
                && removeAndCloseIfIdle(entry)) {
            return null;
        }
        return entry != null ? entry.tryAcquire() : null;
    }

    private void evictExpiredIdleEntries() {
        synchronized (evictionLock) {
            long now = nowNanos();
            List<Throwable> failures = new ArrayList<>();
            entries.values().stream()
                    .filter(entry -> entry.isExpiredIdle(now, expireAfterAccessNanos))
                    .sorted(Comparator.comparingLong(Entry::lastAccessNanos))
                    .forEach(entry -> closeCandidate(entry, failures));
            throwCombined(failures, "Failed to close expired Harness cache entries");
        }
    }

    private void evictOneLeastRecentlyUsedIdleEntry() {
        synchronized (evictionLock) {
            entries.values().stream()
                    .filter(Entry::isIdle)
                    .min(Comparator.comparingLong(Entry::lastAccessNanos))
                    .ifPresent(this::removeAndCloseIfIdle);
        }
    }

    private boolean removeAndCloseIfIdle(Entry entry) {
        if (!entry.markRemovedIfIdle()) {
            return false;
        }
        entries.remove(entry.key, entry);
        entry.closeOnce();
        return true;
    }

    private void discardFailedCreation(Entry entry, Throwable originalFailure) {
        entries.remove(entry.key, entry);
        if (!entry.markRemovedIfIdle()) {
            originalFailure.addSuppressed(new IllegalStateException(
                    "Failed Harness cache entry retained an active lease"));
            return;
        }
        try {
            entry.closeOnce();
        } catch (Throwable closeFailure) {
            if (closeFailure != originalFailure) {
                originalFailure.addSuppressed(closeFailure);
            }
        }
    }

    private Mono<Void> closeAllIdleEntries() {
        List<Entry> snapshot = List.copyOf(entries.values());
        return Flux.fromIterable(snapshot)
                .flatMap(this::closeIdleEntryOnModel, Math.max(1, snapshot.size()))
                .then(Mono.<Void>fromRunnable(this::throwDrainCloseFailures));
    }

    private Mono<Void> closeIdleEntryOnModel(Entry entry) {
        return Mono.create(sink -> {
            Runnable cleanup = () -> {
                try {
                    if (entry.markRemovedIfIdle()) {
                        entries.remove(entry.key, entry);
                        entry.closeOnce();
                    }
                } catch (Throwable failure) {
                    drainCloseFailures.add(failure);
                    log.error("Harness cache entry cleanup failed", failure);
                } finally {
                    sink.success();
                }
            };
            try {
                modelScheduler.schedule(cleanup);
            } catch (RuntimeException schedulingFailure) {
                startEmergencyCleanup(cleanup, "Harness entry", schedulingFailure);
            }
        });
    }

    private void throwDrainCloseFailures() {
        List<Throwable> failures = new ArrayList<>();
        Throwable delayed;
        while ((delayed = drainCloseFailures.poll()) != null) {
            failures.add(delayed);
        }
        throwCombined(failures, "Failed to close Harness cache");
    }

    private Mono<Void> cleanupAfterDrainTimeout(BusinessException timeoutFailure) {
        return Mono.defer(() -> {
            closeAllIdleEntries().subscribe(
                    ignored -> { },
                    failure -> log.error("Harness idle cleanup failed after drain timeout", failure));
            return Mono.error(timeoutFailure);
        });
    }

    private void closeCandidate(Entry entry, List<Throwable> failures) {
        try {
            removeAndCloseIfIdle(entry);
        } catch (Throwable failure) {
            failures.add(failure);
        }
    }

    private Mono<Void> awaitZeroActiveLeases() {
        return Mono.defer(() -> {
            Sinks.Empty<Void> waiter = Sinks.empty();
            if (!zeroLeaseWaiter.compareAndSet(null, waiter)) {
                waiter = zeroLeaseWaiter.get();
            }
            if (activeLeases.get() == 0) {
                zeroLeaseWaiter.compareAndSet(waiter, null);
                return Mono.empty();
            }
            return waiter.asMono();
        });
    }

    private void leaseAcquired() {
        activeLeases.incrementAndGet();
    }

    private void leaseReleased(Entry entry, boolean idle) {
        Throwable closeFailure = null;
        if (idle && draining.get()) {
            try {
                removeAndCloseIfIdle(entry);
            } catch (Throwable failure) {
                drainCloseFailures.add(failure);
                closeFailure = failure;
            }
        }
        int count = activeLeases.decrementAndGet();
        if (count < 0) {
            throw new IllegalStateException("Harness active lease count became negative");
        }
        if (count == 0) {
            Sinks.Empty<Void> waiter = zeroLeaseWaiter.getAndSet(null);
            if (waiter != null) {
                waiter.tryEmitEmpty();
            }
        }
        if (closeFailure != null) {
            throwUnchecked(closeFailure, "Failed to close released Harness cache entry");
        }
    }

    private void awaitInFlightCreates() {
        lifecycleLock.writeLock().lock();
        try {
            // Acquiring the write lock is the barrier: all pre-drain creators have left.
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }

    private BusinessException shuttingDown() {
        return new BusinessException(503, "HARNESS_SHUTTING_DOWN");
    }

    private static Duration requirePositive(Duration value, String name) {
        Objects.requireNonNull(value, name + " must not be null");
        if (value.isZero() || value.isNegative()) {
            throw new IllegalArgumentException(name + " must be greater than zero");
        }
        return value;
    }

    private static long requirePositiveNanos(Duration value, String name) {
        requirePositive(value, name);
        try {
            return value.toNanos();
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException(name + " is too large", overflow);
        }
    }

    private long nowNanos() {
        return nanoTime.getAsLong();
    }

    private static void throwCombined(List<Throwable> failures, String message) {
        if (failures.isEmpty()) {
            return;
        }
        Throwable first = failures.getFirst();
        failures.stream().skip(1).filter(failure -> failure != first).forEach(first::addSuppressed);
        throwUnchecked(first, message);
    }

    private static void throwUnchecked(Throwable failure, String message) {
        if (failure instanceof RuntimeException runtimeException) {
            throw runtimeException;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        throw new IllegalStateException(message, failure);
    }

    private KeyLock retainKeyLock(AgentKernelKey key) {
        return creationLocks.compute(key, (ignored, existing) -> {
            KeyLock retained = existing != null ? existing : new KeyLock();
            retained.references++;
            return retained;
        });
    }

    private void releaseKeyLock(AgentKernelKey key, KeyLock keyLock) {
        creationLocks.computeIfPresent(key, (ignored, existing) -> {
            if (existing != keyLock) {
                return existing;
            }
            existing.references--;
            return existing.references == 0 ? null : existing;
        });
    }

    private static final class KeyLock {
        private int references;
    }

    private record AcquireAttempt(HarnessLease lease) implements AutoCloseable {
        @Override
        public void close() {
            if (lease != null) {
                lease.close();
            }
        }
    }

    private final class Entry {
        private final AgentKernelKey key;
        private final AgentKernelResource resource;
        private final AtomicBoolean closed = new AtomicBoolean();
        private int leaseCount;
        private boolean removed;
        private long lastAccessNanos = nowNanos();

        private Entry(AgentKernelKey key, AgentKernelResource resource) {
            this.key = key;
            this.resource = Objects.requireNonNull(resource, "factory returned null resource");
        }

        private synchronized HarnessLease tryAcquire() {
            if (removed) {
                return null;
            }
            HarnessLease lease = new HarnessLease(resource, this::releaseLease);
            leaseCount++;
            try {
                lastAccessNanos = nowNanos();
                leaseAcquired();
                return lease;
            } catch (Throwable failure) {
                leaseCount--;
                throw failure;
            }
        }

        private void releaseLease() {
            boolean idle;
            synchronized (this) {
                if (leaseCount <= 0) {
                    throw new IllegalStateException("Harness entry has no active lease to release");
                }
                leaseCount--;
                lastAccessNanos = nowNanos();
                idle = leaseCount == 0;
            }
            leaseReleased(this, idle);
        }

        private synchronized boolean isIdle() {
            return !removed && leaseCount == 0;
        }

        private synchronized boolean isExpiredIdle(long now, long ttlNanos) {
            return !removed && leaseCount == 0 && now - lastAccessNanos >= ttlNanos;
        }

        private synchronized boolean markRemovedIfIdle() {
            if (removed || leaseCount != 0) {
                return false;
            }
            removed = true;
            return true;
        }

        private synchronized long lastAccessNanos() {
            return lastAccessNanos;
        }

        private void closeOnce() {
            if (!closed.compareAndSet(false, true)) {
                return;
            }
            try {
                resource.close();
            } finally {
                capacityPermits.release();
            }
        }
    }
}
