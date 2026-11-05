package com.stonewu.fusion.service.media.deletion;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.entity.media.MediaBlobEntity;
import com.stonewu.fusion.entity.media.MediaPurgeEntity;
import com.stonewu.fusion.entity.media.MediaResourceEntity;
import com.stonewu.fusion.mapper.media.MediaBlobMapper;
import com.stonewu.fusion.mapper.media.MediaPurgeMapper;
import com.stonewu.fusion.mapper.media.MediaResourceMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;

@Service
@RequiredArgsConstructor
public class MediaPurgeTransactions {
    private final MediaPurgeMapper purges;
    private final MediaBlobMapper blobs;
    private final MediaResourceMapper resources;

    public List<MediaPurgeEntity> pending() {
        return purges.selectPage(new Page<MediaPurgeEntity>(1, 10, false), Wrappers.<MediaPurgeEntity>lambdaQuery()
                .eq(MediaPurgeEntity::getState, "PENDING")
                .le(MediaPurgeEntity::getNextAttemptAt, LocalDateTime.now(ZoneOffset.UTC))
                .orderByAsc(MediaPurgeEntity::getNextAttemptAt)).getRecords();
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public boolean mayDelete(String id) {
        var job = purges.selectById(id);
        return job != null && job.getState().equals("PENDING") && "ORPHAN".equals(blobs.lockState(id))
                && !resources.exists(Wrappers.<MediaResourceEntity>lambdaQuery()
                .eq(MediaResourceEntity::getBlobId, id).in(MediaResourceEntity::getState, "READY", "STAGING"));
    }

    @Transactional
    @CacheEvict(value = "mediaResource", allEntries = true)
    public void complete(String id) {
        blobs.update(Wrappers.<MediaBlobEntity>lambdaUpdate().eq(MediaBlobEntity::getId, id)
                .eq(MediaBlobEntity::getState, "ORPHAN").set(MediaBlobEntity::getState, "PURGED")
                .set(MediaBlobEntity::getKeysetBytes, new byte[0]));
        purges.update(Wrappers.<MediaPurgeEntity>lambdaUpdate().eq(MediaPurgeEntity::getBlobId, id)
                .set(MediaPurgeEntity::getState, "PURGED"));
    }

    @Transactional
    @CacheEvict(value = "mediaResource", allEntries = true)
    public void defer(String id) {
        var job = purges.selectById(id);
        if (job == null || !job.getState().equals("PENDING")) {
            return;
        }
        int attempts = job.getAttempts() + 1;
        purges.update(Wrappers.<MediaPurgeEntity>lambdaUpdate().eq(MediaPurgeEntity::getBlobId, id)
                .eq(MediaPurgeEntity::getState, "PENDING")
                .set(MediaPurgeEntity::getAttempts, attempts)
                .set(MediaPurgeEntity::getNextAttemptAt, LocalDateTime.now(ZoneOffset.UTC)
                        .plusSeconds(Math.min(3600, 30L << Math.min(attempts, 7)))));
    }
}
