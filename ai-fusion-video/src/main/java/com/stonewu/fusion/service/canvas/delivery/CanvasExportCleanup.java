package com.stonewu.fusion.service.canvas.delivery;

import com.stonewu.fusion.service.media.MediaBlobStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

/**
 * 未发布交付文件使用已有状态时间及事件审计，已发布交付始终保留。
 */
@Component
public class CanvasExportCleanup {
    private final CanvasExportTransactions transactions;
    private final MediaBlobStore blobs;
    private final int days;

    public CanvasExportCleanup(CanvasExportTransactions transactions, MediaBlobStore blobs,
                               @Value("${app.media.cleanup.retention-days:30}") int days) {
        if (days < 1 || days > 3650) {
            throw new IllegalArgumentException("交付孤儿保留期必须为 1–3650 天");
        }
        this.transactions = transactions;
        this.blobs = blobs;
        this.days = days;
    }

    public boolean runOne() {
        var candidate = transactions.expiredOrphan(days);
        if (candidate.isEmpty()) {
            return false;
        }
        var blob = candidate.get();
        try {
            if (blobs.purgeDelivery(blob, () -> transactions.canPurge(blob, days))) {
                transactions.purged(blob);
            }
        } catch (Exception failure) {
            if (!Thread.currentThread().isInterrupted()) {
                transactions.purgeRetry(blob);
            }
        }
        return true;
    }
}
