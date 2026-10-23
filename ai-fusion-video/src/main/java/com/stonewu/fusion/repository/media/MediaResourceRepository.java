package com.stonewu.fusion.repository.media;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.media.model.*;
import com.stonewu.fusion.service.media.preview.MediaPreviewData.Parent;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.media.MediaResourceEntity;
import com.stonewu.fusion.entity.media.MediaBlobEntity;
import com.stonewu.fusion.entity.media.MediaEventEntity;
import com.stonewu.fusion.entity.media.MediaPreviewJobEntity;
import com.stonewu.fusion.entity.media.MediaSourceEntity;
import com.stonewu.fusion.mapper.media.MediaResourceMapper;
import com.stonewu.fusion.mapper.media.MediaBlobMapper;
import com.stonewu.fusion.mapper.media.MediaEventMapper;
import com.stonewu.fusion.mapper.media.MediaPreviewJobMapper;
import com.stonewu.fusion.mapper.media.MediaSourceMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Repository
@RequiredArgsConstructor
public class MediaResourceRepository {
    private final MediaResourceMapper resources;
    private final MediaBlobMapper blobs;
    private final MediaEventMapper events;
    private final MediaPreviewJobMapper previews;
    private final MediaSourceMapper sources;
    private final CanvasJson json;

    public record Row(MediaResource resource, String requestHash, String blobId, long epoch,
                      LocalDateTime leaseUntil, Long actorTeamId) {
    }

    public Optional<Row> request(long actor, String requestId) {
        return Optional.ofNullable(resources.lockRequest(actor, requestId)).map(this::row);
    }

    public Optional<Row> source(MediaSourceReference source, boolean lock) {
        return Optional.ofNullable(
                        resources.source(source.type().code(), Long.toString(source.id()), source.version(), source.field(),
                                lock))
                .map(this::row);
    }

    public void bindSource(String resourceId, MediaSourceReference source) {
        var row = new MediaSourceEntity();
        row.setResourceId(resourceId);
        row.setSourceType(source.type().code());
        row.setSourceId(Long.toString(source.id()));
        row.setSourceVersion(source.version());
        row.setSourceField(source.field());
        sources.insert(row);
    }

    public List<MediaSourceReference> sources(String resourceId) {
        return sources.selectList(new LambdaQueryWrapper<MediaSourceEntity>()
                        .eq(MediaSourceEntity::getResourceId, resourceId)).stream()
                .map(row -> new MediaSourceReference(MediaSourceType.fromCode(row.getSourceType()),
                        Long.parseLong(row.getSourceId()), row.getSourceVersion(), row.getSourceField())).toList();
    }

    public Row get(String id, boolean lock) {
        if (id == null || !id.matches("[0-9a-fA-F-]{36}")) {
            throw MediaException.invalid("媒体标识无效");
        }
        return Optional.ofNullable(lock ? resources.lock(id) : resources.selectById(id)).map(this::row)
                .orElseThrow(() -> new MediaException(404, "MEDIA_NOT_FOUND", "媒体资源不存在"));
    }

    public MediaBlob blob(String id) {
        return Optional.ofNullable(blobs.selectOne(new LambdaQueryWrapper<MediaBlobEntity>()
                        .eq(MediaBlobEntity::getId, id).in(MediaBlobEntity::getState, "STAGING", "READY")))
                .map(this::blobRow).orElseThrow(() -> MediaException.unavailable("媒体存储对象不可用"));
    }

    /**
     * 仅供回收任务定位孤儿密文；正常内容读取仍须通过 blob 的可用状态检查。
     */
    public MediaBlob orphanBlob(String id) {
        return Optional.ofNullable(blobs.selectOne(new LambdaQueryWrapper<MediaBlobEntity>()
                        .eq(MediaBlobEntity::getId, id).eq(MediaBlobEntity::getState, "ORPHAN")))
                .map(this::blobRow).orElseThrow(() -> MediaException.unavailable("待清理媒体对象状态已变化"));
    }

    /**
     * 保留判断只查目标媒体，避免为一次 EXISTS 展开全部历史文档与媒体的联合视图。不可缓存。
     */
    public Optional<String> retentionReference(String mediaId) {
        String direct = resources.directRetention(mediaId);
        return direct != null ? Optional.of(direct) : Optional.ofNullable(resources.documentRetention(mediaId));
    }

    private MediaBlob blobRow(MediaBlobEntity row) {
        return new MediaBlob(row.getId(), row.getResourceId(),
                row.getStorageConfigId(), row.getBackend(), row.getObjectKey(),
                row.getKeysetBytes(), row.getStoredBytes());
    }

    public List<MediaResource> list(Long projectId, int ownerType, long ownerId, int limit, int offset) {
        return resources.readyResources(projectId, ownerType, ownerId, limit, offset)
                .stream().map(this::row).map(Row::resource).toList();
    }

    public Optional<Parent> previewParent(String resourceId) {
        return Optional.ofNullable(previews.selectOne(new LambdaQueryWrapper<MediaPreviewJobEntity>()
                        .select(MediaPreviewJobEntity::getSourceResourceId, MediaPreviewJobEntity::getSourceVersion)
                        .eq(MediaPreviewJobEntity::getOutputResourceId, resourceId)))
                .map(row -> new Parent(row.getSourceResourceId(), row.getSourceVersion()));
    }

    public void event(MediaResource resource, String type, long actor) {
        events.insert(MediaEventEntity.builder().eventId(UUID.randomUUID().toString()).resourceId(resource.id())
                .resourceVersion(resource.version()).eventType(type).actorUserId(actor)
                .projectId(resource.projectId()).payloadJson(json.write(resource)).build());
    }

    private Row row(MediaResourceEntity row) {
        var facts = new MediaFacts(MediaKind.valueOf(row.getMediaKind()), row.getMimeType(),
                row.getByteSize(), row.getContentHash(), row.getWidth(),
                row.getHeight(), row.getDurationMs(),
                row.getHasAudio());
        var resource = new MediaResource(row.getId(), row.getProjectId(),
                row.getOwnerType(), row.getOwnerId(), row.getCreatedBy(),
                row.getOriginalName(),
                facts, row.getState(), row.getVersion(), row.getLastError(),
                row.getCreatedAt(), row.getUpdatedAt());
        return new Row(resource, row.getRequestHash(), row.getBlobId(), row.getIngestEpoch(),
                row.getLeaseUntil(), row.getActorTeamId());
    }
}
