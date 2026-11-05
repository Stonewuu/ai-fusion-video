package com.stonewu.fusion.service.media.deletion;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.media.MediaBlobStore;
import com.stonewu.fusion.service.media.MediaIoBoundary;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import jakarta.annotation.PreDestroy;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** 永久删除已获用户确认；历史记录保留删除状态，不再阻止相应文件清理。 */
@Component
@Slf4j
@RequiredArgsConstructor
public class MediaPurgeWorker {
    private final MediaPurgeTransactions jobs;
    private final MediaResourceRepository resources;
    private final MediaBlobStore storage;
    private final AtomicBoolean busy = new AtomicBoolean();
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();

    @Scheduled(fixedDelay = 30000)
    public void dispatch() {
        if (!busy.compareAndSet(false, true)) {
            return;
        }
        try {
            executor.execute(() -> {
                try {
                    runPending();
                } finally {
                    busy.set(false);
                }
            });
        } catch (RuntimeException failure) {
            busy.set(false);
            if (!executor.isShutdown()) {
                throw failure;
            }
        }
    }

    public void runPending() {
        MediaIoBoundary.requireOutsideTransaction();
        for (var job : jobs.pending()) {
            try {
                if (storage.purge(resources.orphanBlob(job.getBlobId()), () -> jobs.mayDelete(job.getBlobId()))) {
                    jobs.complete(job.getBlobId());
                } else {
                    jobs.defer(job.getBlobId());
                }
            } catch (Exception failure) {
                jobs.defer(job.getBlobId());
                log.warn("媒体文件删除待重试 [blobId={}, failureType={}]",
                        job.getBlobId(), failure.getClass().getSimpleName());
            }
        }
    }

    @PreDestroy
    public void close() {
        executor.shutdownNow();
    }
}
