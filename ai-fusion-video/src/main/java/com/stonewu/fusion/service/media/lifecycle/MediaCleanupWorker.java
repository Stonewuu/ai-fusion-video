package com.stonewu.fusion.service.media.lifecycle;

import com.stonewu.fusion.service.canvas.delivery.CanvasExportCleanup;
import com.stonewu.fusion.service.media.MediaBlobStore;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
@Component
@RequiredArgsConstructor
public class MediaCleanupWorker {
    private final AgentWorkspaceMediaIndexer workspace;
    private final MediaOrphanCleanup jobs;
    private final MediaBlobStore blobs;
    private final CanvasExportCleanup exports;
    private final String owner = UUID.randomUUID().toString();
    private final AtomicBoolean busy = new AtomicBoolean();
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();

    @Scheduled(fixedDelayString = "${app.media.cleanup.interval-ms:60000}", initialDelayString = "${app.media.cleanup.initial-delay-ms:60000}")
    public void dispatch() {
        if (!busy.compareAndSet(false, true)) {
            return;
        }
        try {
            executor.execute(() -> {
                try {
                    if (workspace.indexBatch()) {
                        jobs.discover();
                        for (int index = 0; index < 10 && runOne(); index++) {
                        }
                    }
                    for (int index = 0; index < 10 && exports.runOne(); index++) {
                    }
                } catch (RuntimeException failure) {
                    log.warn("媒体引用检查或孤儿回收暂未完成 [failureType={}]", failure.getClass().getSimpleName());
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

    public boolean runOne() {
        var next = jobs.claim(owner);
        if (next.isEmpty()) {
            return false;
        }
        var claim = next.get();
        try {
            if (blobs.purge(jobs.blob(claim), () -> jobs.mayDelete(claim))) {
                jobs.complete(claim);
            } else {
                jobs.defer(claim, true);
            }
        } catch (Exception failure) {
            if (Thread.currentThread().isInterrupted()) {
                return true;
            }
            jobs.defer(claim, false);
            log.warn("媒体孤儿文件清理待重试 [blobId={}, failureType={}]", claim.blobId(),
                    failure.getClass().getSimpleName());
        }
        return true;
    }

    @PreDestroy
    public void close() {
        executor.shutdownNow();
    }
}
