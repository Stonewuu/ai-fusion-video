package com.stonewu.fusion.service.media.importing;

import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.media.MediaSourceImportService;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaJobOrigin;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.dao.DataAccessException;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

@Component
@RequiredArgsConstructor
@Slf4j
public class MediaImportWorker {
    private final MediaImportJobs imports;
    private final MediaJobRepository jobs;
    private final MediaSourceImportService sources;
    private final String owner = "media-import-" + UUID.randomUUID();
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private final Semaphore slots = new Semaphore(2);
    private final Set<MediaJob.Claim> active = ConcurrentHashMap.newKeySet();

    @Scheduled(fixedDelay = 500)
    public void dispatch() {
        while (slots.tryAcquire()) {
            Optional<MediaJob> claimed;
            try {
                claimed = imports.claim(owner);
            } catch (RuntimeException failure) {
                slots.release();
                throw failure;
            }
            if (claimed.isEmpty()) {
                slots.release();
                return;
            }
            var claim = claimed.get().claim();
            active.add(claim);
            try {
                executor.execute(() -> {
                    try {
                        process(claim);
                    } finally {
                        active.remove(claim);
                        slots.release();
                    }
                });
            } catch (RuntimeException failure) {
                active.remove(claim);
                slots.release();
                if (!executor.isShutdown()) {
                    throw failure;
                }
                return;
            }
        }
    }

    @Scheduled(fixedDelay = 20000)
    public void heartbeat() {
        active.forEach(jobs::heartbeat);
    }

    public boolean runOne() {
        var claimed = imports.claim(owner);
        if (claimed.isEmpty()) {
            return false;
        }
        var claim = claimed.get().claim();
        active.add(claim);
        try {
            process(claim);
        } finally {
            active.remove(claim);
        }
        return true;
    }

    private void process(MediaJob.Claim claim) {
        MediaImportJobs.Work work = null;
        try {
            work = imports.start(claim);
            if (work == null) {
                return;
            }
            var result = sources.importJobSource(work.source().snapshot(), work.job().context(),
                    new MediaJobOrigin(claim), () -> imports.checked(claim));
            imports.record(claim, work.index(), result.id(), null, null);
        } catch (Exception failure) {
            if (Thread.currentThread().isInterrupted()) {
                return;
            }
            if (failure instanceof DataAccessException) {
                log.warn("导入状态尚未保存，将由原任务租约恢复 [jobId={}]", claim.id(), failure);
                return;
            }
            try {
                if (work == null || failure instanceof MediaException media && media.getErrorCode()
                        .equals("MEDIA_JOB_FENCED")) {
                    imports.stop(claim, "导入已取消、到期或原访问权限已变化；已导入素材保留");
                } else if (failure instanceof MediaException media && media.getErrorCode()
                        .equals("MEDIA_INGEST_IN_PROGRESS")) {
                    imports.waitForIngest(claim);
                } else {
                    String code = failure instanceof MediaException media ? media.getErrorCode() : "MEDIA_IMPORT_FAILED";
                    String message = failure instanceof MediaException || failure instanceof CanvasException ? failure.getMessage()
                            : "读取或保存来源文件失败，请核对来源后重试此项";
                    log.warn("来源素材导入失败 [jobId={}, index={}]", claim.id(), work.index(), failure);
                    imports.record(claim, work.index(), null, code, message);
                }
            } catch (MediaException | CanvasException stopped) {
                try {
                    imports.stop(claim, "原访问范围或执行状态已变化，停止后续导入");
                } catch (RuntimeException unavailable) {
                    log.warn("导入收尾尚未保存，将由原租约恢复 [jobId={}]", claim.id(), unavailable);
                }
            } catch (RuntimeException unavailable) {
                log.warn("导入状态尚未保存，将由原任务恢复 [jobId={}]", claim.id(), unavailable);
            }
        }
    }

    @PreDestroy
    public void stop() {
        executor.shutdownNow();
    }
}
