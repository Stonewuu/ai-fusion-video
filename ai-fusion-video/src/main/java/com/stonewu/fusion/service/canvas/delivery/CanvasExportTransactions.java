package com.stonewu.fusion.service.canvas.delivery;

import com.stonewu.fusion.mapper.canvas.CanvasExportBlobMapper;
import com.stonewu.fusion.mapper.canvas.CanvasExportMapper;
import com.stonewu.fusion.entity.canvas.CanvasExportBlobEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.media.model.MediaBlob;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.Map;
import java.util.Optional;

import static com.stonewu.fusion.service.canvas.delivery.CanvasExportData.*;

@Service
@RequiredArgsConstructor
public class CanvasExportTransactions {
    private final CanvasExportRepository jobs;
    private final CanvasExportSources sources;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final CanvasExportBlobMapper blobs;
    private final CanvasExportMapper exports;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Claim claim(String id) {
        return jobs.claim(id);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Job check(Claim claim) {
        var job = jobs.leased(claim, true);
        active(job);
        jobs.heartbeat(claim);
        return job;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public MediaBlob reserve(Claim claim) {
        var job = jobs.leased(claim, true);
        active(job);
        return jobs.reserve(job);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasExport", allEntries = true)
    public void complete(Claim claim, MediaBlob blob, long storedBytes, long size, String hash) {
        var job = jobs.leased(claim, true);
        active(job);
        jobs.complete(job, blob, storedBytes, size, hash);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasExport", allEntries = true)
    public void fail(Claim claim, String error) {
        var job = jobs.get(claim.id(), true);
        if (job.terminal() || job.epoch() != claim.epoch() || !claim.worker().equals(job.worker())) {
            return;
        }
        jobs.finishState(job, job.cancelRequested() ? "CANCELLED" : "FAILED", job.cancelRequested() ? null : error);
    }

    public Optional<MediaBlob> expiredOrphan(int days) {
        return Optional.ofNullable(blobs.expiredOrphan(-days)).map(CanvasExportRepository::blob);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public boolean canPurge(MediaBlob blob, int days) {
        var job = jobs.get(blob.resourceId(), true);
        if (blob.id().equals(job.blobId())) {
            return false;
        }
        return blobs.canPurge(blob.id(), blob.resourceId(), -days) == 1;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasExport", allEntries = true)
    public void purged(MediaBlob blob) {
        jobs.get(blob.resourceId(), true);
        if (blobs.purged(blob.id()) == 1) {
            jobs.event(blob.resourceId(), null, "FILE_PURGED", Map.of("blobId", blob.id()));
        }
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasExport", allEntries = true)
    public void purgeRetry(MediaBlob blob) {
        jobs.get(blob.resourceId(), true);
        if (blobs.selectCount(
                Wrappers.<CanvasExportBlobEntity>lambdaQuery().eq(CanvasExportBlobEntity::getId, blob.id())
                        .eq(CanvasExportBlobEntity::getState, "ORPHANED")) == 1) {
            jobs.event(blob.resourceId(), null, "FILE_PURGE_RETRY",
                    Map.of("blobId", blob.id(), "message", "未完成文件清理，将在一小时后重试"));
        }
    }

    private void active(Job job) {
        if (job.cancelRequested()) {
            throw conflict("交付任务已取消");
        }
        if (job.createdAt().plusSeconds(7200).isBefore(Instant.now())) {
            throw conflict("交付任务已超过两小时，请缩小范围重新导出");
        }
        sources.authorize(job.canvasId(), job.manifest(), job.context().actor());
        if (job.nodeRunId() != null) {
            access.requireExecute(documents.require(job.canvasId(), false), job.context().actor());
            int valid = exports.activeNodeRun(job.nodeRunId());
            if (valid != 1) {
                throw conflict("所属画布运行已取消或结束");
            }
        }
    }
}
