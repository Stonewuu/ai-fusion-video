package com.stonewu.fusion.service.canvas.delivery;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.media.MediaBlobStore;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

import static com.stonewu.fusion.service.canvas.delivery.CanvasExportData.Claim;

@Component
@Slf4j
@RequiredArgsConstructor
@ConditionalOnProperty(name = "app.canvas.exports.enabled", havingValue = "true", matchIfMissing = true)
public class CanvasExportWorker {
    private final CanvasExportRepository jobs;
    private final CanvasExportTransactions transactions;
    private final CanvasExportWriter writer;
    private final MediaBlobStore blobs;
    private final ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor();
    private final Semaphore slots = new Semaphore(2);
    private final Map<String, Claim> active = new ConcurrentHashMap<>();

    @Scheduled(fixedDelay = 1000)
    public void tick() {
        for (String id : jobs.pending()) {
            if (!slots.tryAcquire()) {
                return;
            }
            try {
                var claim = transactions.claim(id);
                if (claim == null) {
                    slots.release();
                    continue;
                }
                active.put(id, claim);
                pool.submit(() -> {
                    try {
                        execute(claim);
                    } finally {
                        active.remove(id, claim);
                        slots.release();
                    }
                });
            } catch (RuntimeException failure) {
                slots.release();
                log.warn("交付任务暂未领取 [id={}]", id, failure);
            }
        }
    }

    public void execute(Claim claim) {
        Path output = null;
        try {
            var job = transactions.check(claim);
            output = Files.createTempFile("afv-export-", ".part");
            writer.write(job.manifest(), job.format(), job.context().actor(), output, () -> transactions.check(claim));
            String hash = CanvasExportWriter.hash(output);
            var blob = transactions.reserve(claim);
            long storedBytes = blobs.writeDelivery(blob, output);
            transactions.complete(claim, blob, storedBytes, Files.size(output), hash);
        } catch (Exception failure) {
            log.warn("交付构建未完成 [id={}]", claim.id(), failure);
            String error = failure instanceof BusinessException ? failure.getMessage() : "交付文件构建失败，请检查存储状态后重新导出";
            try {
                transactions.fail(claim, error.substring(0, Math.min(error.length(), 1000)));
            } catch (RuntimeException retry) {
                log.warn("交付终态暂未保存，将由租约恢复 [id={}]", claim.id(), retry);
            }
        } finally {
            if (output != null) {
                try {
                    Files.deleteIfExists(output);
                } catch (Exception cleanup) {
                    log.warn("交付临时文件暂未清理 [id={}]", claim.id());
                }
            }
        }
    }

    @Scheduled(fixedDelay = 15000)
    public void heartbeat() {
        active.values().forEach(claim -> {
            try {
                transactions.check(claim);
            } catch (RuntimeException stopped) {
                log.debug("交付不再续租 [id={}]", claim.id());
            }
        });
    }

    @PreDestroy
    public void close() {
        pool.shutdownNow();
    }
}
