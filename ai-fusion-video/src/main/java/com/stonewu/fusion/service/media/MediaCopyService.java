package com.stonewu.fusion.service.media;

import com.stonewu.fusion.mapper.media.MediaResourceMapper;
import com.stonewu.fusion.mapper.media.MediaBlobMapper;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.UUID;

/**
 * 复制元数据与归属，共享已完成的不可变密文；读取副本仅检查副本自身的当前权限。
 */
@Service
@RequiredArgsConstructor
public class MediaCopyService {
    private final MediaResourceRepository resources;
    private final MediaAccessPolicy access;
    private final CanvasJson json;
    private final MediaResourceMapper media;
    private final MediaBlobMapper blobs;

    private record Scope(String sourceId, int sourceVersion, Long projectId, CanvasActor actor) {
    }

    @Transactional(propagation = Propagation.MANDATORY)
    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaResource copy(String sourceId, int sourceVersion, Long projectId, String operationId,
                              CanvasActor actor) {
        String uuid = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";
        if (operationId == null || !operationId.matches(uuid) || sourceId == null || !sourceId.matches(
                uuid) || sourceVersion < 0) {
            throw MediaException.invalid("媒体复制来源或操作标识无效");
        }
        var context = ResourceCommandContext.user(actor, null);
        var owner = access.creation(projectId, context);
        String requestId = "copy:" + operationId + ":" + sourceId;
        String hash = json.hash(new Scope(sourceId, sourceVersion, projectId, actor));
        var replay = resources.request(actor.userId(), requestId);
        if (replay.isPresent()) {
            if (!replay.get().requestHash().equals(hash)) {
                throw new MediaException(409, "MEDIA_COPY_CONFLICT", "媒体复制请求已用于其他内容");
            }
            access.ready(replay.get().resource(), actor);
            return replay.get().resource();
        }
        var source = resources.get(sourceId, true);
        access.ready(source.resource(), actor);
        if (source.resource().version() != sourceVersion) {
            throw new MediaException(409, "MEDIA_COPY_CHANGED", "来源媒体版本已经变化，请重新预览复制内容");
        }
        if (blobs.lockReady(source.blobId()) == null) {
            throw MediaException.unavailable("来源媒体尚无可复制的完整文件");
        }
        String id = UUID.randomUUID().toString();
        media.copy(id, projectId, owner.type(), owner.id(), actor.userId(), actor.teamId(), requestId, hash, sourceId);
        media.recordCopy(id, sourceId, sourceVersion, actor.userId());
        var copied = resources.get(id, false).resource();
        resources.event(copied, "media.copied", actor.userId());
        return copied;
    }
}
