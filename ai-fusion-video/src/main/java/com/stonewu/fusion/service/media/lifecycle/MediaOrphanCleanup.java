package com.stonewu.fusion.service.media.lifecycle;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.media.model.MediaBlob;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.annotation.CacheEvict;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.stonewu.fusion.entity.media.MediaBlobEntity;
import com.stonewu.fusion.entity.media.MediaResourceEntity;
import com.stonewu.fusion.entity.ai.AgentWorkspaceEntry;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceEntryMapper;
import com.stonewu.fusion.mapper.media.MediaBlobCleanupMapper;
import com.stonewu.fusion.mapper.media.MediaBlobMapper;
import com.stonewu.fusion.mapper.media.MediaResourceMapper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * 只回收接收失败的孤儿密文；READY 文件及其任一有效副本均不进入物理删除。
 */
@Service
public class MediaOrphanCleanup {
    public record Claim(String blobId, String owner, long epoch, int attempts) {
    }

    private final MediaBlobCleanupMapper cleanup;
    private final MediaBlobMapper blobs;
    private final MediaResourceMapper resources;
    private final AgentWorkspaceEntryMapper workspace;
    private final MediaResourceRepository media;
    private final int retentionDays;

    public MediaOrphanCleanup(MediaBlobCleanupMapper cleanup, MediaBlobMapper blobs, MediaResourceMapper resources,
                              AgentWorkspaceEntryMapper workspace, MediaResourceRepository media,
                              @Value("${app.media.cleanup.retention-days:30}") int retentionDays) {
        if (retentionDays < 1 || retentionDays > 3650) {
            throw new IllegalArgumentException("媒体孤儿保留期必须为 1–3650 天");
        }
        this.cleanup = cleanup;
        this.blobs = blobs;
        this.resources = resources;
        this.workspace = workspace;
        this.media = media;
        this.retentionDays = retentionDays;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaCleanup", allEntries = true)
    public void discover() {
        cleanup.discover(retentionDays);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaCleanup", allEntries = true)
    public Optional<Claim> claim(String owner) {
        if (owner == null || !owner.matches("[0-9a-f-]{36}")) {
            throw new IllegalArgumentException("清理工作者标识无效");
        }
        if (!workspaceIndexed()) {
            return Optional.empty();
        }
        String id = cleanup.next();
        if (id == null) {
            return Optional.empty();
        }
        // 这里只领取任务；物理删除前再按任务→文件的顺序锁定并复核。
        var blob = blobs.selectById(id);
        if (blob == null || !"ORPHAN".equals(blob.getState()) || protectedBlob(id)) {
            block(id, "文件仍被有效媒体或保留记录使用");
            return Optional.empty();
        }
        if (!Boolean.TRUE.equals(cleanup.oldEnough(-retentionDays, id))) {
            block(id, "尚未达到当前配置的保留期限");
            return Optional.empty();
        }
        cleanup.claim(owner, id);
        var row = cleanup.selectById(id);
        return Optional.of(new Claim(id, row.getOwnerId(), row.getEpoch(), row.getAttempts()));
    }

    public MediaBlob blob(Claim claim) {
        return media.orphanBlob(claim.blobId());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public boolean mayDelete(Claim claim) {
        if (!owned(claim) || !workspaceIndexed()) {
            return false;
        }
        return "ORPHAN".equals(blobs.lockState(claim.blobId())) && !protectedBlob(claim.blobId());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = {"mediaCleanup", "mediaResource"}, allEntries = true)
    public void complete(Claim claim) {
        if (!owned(claim)) {
            return;
        }
        blobs.update(Wrappers.<MediaBlobEntity>lambdaUpdate().eq(MediaBlobEntity::getId, claim.blobId())
                .eq(MediaBlobEntity::getState, "ORPHAN").set(MediaBlobEntity::getState, "PURGED")
                .set(MediaBlobEntity::getKeysetBytes, new byte[0]));
        cleanup.purged(claim.blobId());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaCleanup", allEntries = true)
    public void defer(Claim claim, boolean blocked) {
        if (!owned(claim)) {
            return;
        }
        if (blocked) {
            block(claim.blobId(), "保留关系或工作空间索引在清理前变化");
            return;
        }
        int minutes = Math.min(1440, 1 << Math.min(claim.attempts(), 10));
        cleanup.defer(minutes, claim.blobId());
    }

    private boolean owned(Claim claim) {
        var row = cleanup.lockLease(claim.blobId());
        return row != null && Objects.equals(row.ownerId(), claim.owner()) && row.epoch() == claim.epoch()
                && "CLAIMED".equals(row.status()) && Boolean.TRUE.equals(row.valid());
    }

    private boolean workspaceIndexed() {
        return !workspace.exists(Wrappers.<AgentWorkspaceEntry>lambdaQuery()
                .isNull(AgentWorkspaceEntry::getMediaReferencesJson));
    }

    private boolean protectedBlob(String id) {
        for (var resource : resources.selectList(Wrappers.<MediaResourceEntity>lambdaQuery()
                .select(MediaResourceEntity::getId, MediaResourceEntity::getState)
                .eq(MediaResourceEntity::getBlobId, id))) {
            if (List.of("READY", "STAGING").contains(resource.getState())
                    || media.retentionReference(resource.getId()).isPresent()) {
                return true;
            }
        }
        return false;
    }

    private void block(String id, String reason) {
        cleanup.block(reason, id);
    }
}
