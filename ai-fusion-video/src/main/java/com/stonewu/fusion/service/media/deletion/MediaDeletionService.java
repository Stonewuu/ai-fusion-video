package com.stonewu.fusion.service.media.deletion;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.media.MediaAccessTicketEntity;
import com.stonewu.fusion.entity.media.MediaBlobEntity;
import com.stonewu.fusion.entity.media.MediaDeletionEntity;
import com.stonewu.fusion.entity.media.MediaPurgeEntity;
import com.stonewu.fusion.entity.media.MediaResourceEntity;
import com.stonewu.fusion.mapper.media.MediaAccessTicketMapper;
import com.stonewu.fusion.mapper.media.MediaBlobMapper;
import com.stonewu.fusion.mapper.media.MediaDeletionMapper;
import com.stonewu.fusion.mapper.media.MediaPurgeMapper;
import com.stonewu.fusion.mapper.media.MediaResourceMapper;
import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.deletion.MediaDeletionData.*;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@Service
@RequiredArgsConstructor
public class MediaDeletionService {
    private final MediaResourceRepository resources;
    private final MediaResourceMapper media;
    private final MediaBlobMapper blobs;
    private final MediaDeletionMapper deletions;
    private final MediaPurgeMapper purges;
    private final MediaAccessTicketMapper tickets;
    private final MediaAccessPolicy access;
    private final MediaDeletionReferences references;
    private final CanvasJson json;

    @Transactional(readOnly = true, isolation = Isolation.READ_COMMITTED)
    public Preview preview(String id, CanvasActor actor) {
        var row = resources.get(id, false);
        authorize(row.resource(), actor);
        return inspect(row, actor, allReferences(id));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = {"mediaResource", "canvasSnapshot"}, allEntries = true)
    public Result remove(String id, Request request, CanvasActor actor) {
        var row = resources.get(id, true);
        var resource = row.resource();
        authorize(resource, actor);
        var previous = deletions.selectById(id);
        String hash = json.hash(Map.of("request", request, "actor", actor));
        if (previous != null && previous.getRequestId().equals(request.requestId())) {
            if (!previous.getRequestHash().equals(hash)) {
                throw new MediaException(409, "MEDIA_DELETE_REQUEST_CONFLICT", "同一删除请求不能用于其他操作");
            }
            return result(id, Mode.valueOf(previous.getMode()));
        }
        if (previous != null && previous.getMode().equals(request.mode().name())) {
            return result(id, request.mode());
        }
        if (!resource.state().equals("READY") || resource.version() != request.expectedVersion()) {
            throw new MediaException(409, "MEDIA_DELETE_CHANGED", "素材状态已变化，请刷新后重新确认");
        }
        if (request.mode() == Mode.PERMANENT) {
            for (String target : deletionTargets(id)) {
                resources.get(target, true);
            }
        }
        var currentReferences = allReferences(id);
        var preview = inspect(row, actor, currentReferences);
        if (request.mode() == Mode.PERMANENT) {
            if (preview.permanentBlockedReason() != null) {
                throw new MediaException(409, "MEDIA_DELETE_BLOCKED", preview.permanentBlockedReason());
            }
            if (!preview.referenceToken().equals(request.referenceToken())) {
                throw new MediaException(409, "MEDIA_REFERENCES_CHANGED", "素材引用已变化，请重新查看影响范围");
            }
            references.authorize(currentReferences, actor);
            var ids = deletionTargets(id);
            for (String target : ids) {
                references.detach(target, currentReferences, actor);
            }
            if (ids.stream().anyMatch(deletions::activeUse)) {
                throw new MediaException(409, "MEDIA_DELETE_BLOCKED", "确认期间有任务开始使用此素材，请等待任务结束");
            }
            for (String target : ids) {
                markDeleted(target, id, actor);
            }
        }
        var deletion = previous == null ? new MediaDeletionEntity() : previous;
        deletion.setMediaId(id);
        deletion.setMode(request.mode().name());
        deletion.setRequestId(request.requestId());
        deletion.setRequestHash(hash);
        deletion.setActorUserId(actor.userId());
        deletion.setActorTeamId(actor.teamId());
        deletion.setResourceVersion(resource.version());
        deletion.setUpdatedAt(LocalDateTime.now(ZoneOffset.UTC));
        if (previous == null) {
            deletions.insert(deletion);
        } else {
            deletions.updateById(deletion);
        }
        // 仅移除素材库条目由删除表审计，不改变内容版本或占用媒体生命周期事件的版本号。
        if (request.mode() == Mode.PERMANENT) {
            resources.event(resources.get(id, false).resource(), "media.deleted", actor.userId());
        }
        return result(id, request.mode());
    }

    public Result result(String id, Mode mode) {
        if (mode == Mode.LIBRARY) {
            return new Result(id, mode, "RETAINED");
        }
        boolean pending = purges.exists(Wrappers.<MediaPurgeEntity>lambdaQuery()
                .eq(MediaPurgeEntity::getMediaId, id).eq(MediaPurgeEntity::getState, "PENDING"));
        var resource = resources.get(id, false);
        pending = pending || "ORPHAN".equals(blobs.selectById(resource.blobId()).getState());
        boolean shared = deletions.sharedCopies(id, resource.blobId()) > 0;
        return new Result(id, mode, pending ? "PENDING" : shared ? "SHARED" : "PURGED");
    }

    private Preview inspect(MediaResourceRepository.Row row, CanvasActor actor, List<Reference> found) {
        var resource = row.resource();
        long copies = deletions.sharedCopies(resource.id(), row.blobId());
        String blocked = null;
        if (!resource.state().equals("READY")) {
            blocked = "素材当前状态不允许永久删除";
        } else if (deletionTargets(resource.id()).stream().anyMatch(deletions::activeUse)) {
            blocked = "仍有任务正在使用此素材，请等待任务结束或停止任务后再删除";
        } else if (found.stream().anyMatch(item -> item.kind().equals("PRESET"))) {
            blocked = "素材用于系统预设，请先在系统设置中更换预设图片";
        } else {
            try {
                references.authorize(found, actor);
            } catch (BusinessException failure) {
                if (!List.of(403, 404, 409).contains(failure.getCode())) {
                    throw failure;
                }
                blocked = "存在无法修改的项目或画布引用，请先解除引用或恢复画布编辑权限";
            }
        }
        Map<String, Long> counts = new LinkedHashMap<>();
        for (Reference reference : found) {
            String label = reference.kind().equals("BUSINESS_BINDING") ? "项目内容"
                    : reference.kind().equals("CANVAS_NODE") && reference.retention().equals("LIVE") ? "画布节点"
                    : reference.kind().equals("AGENT_WORKSPACE") ? "助手工作空间"
                    : reference.kind().equals("PRESET") ? "系统预设" : "历史记录与任务";
            counts.merge(label, 1L, Long::sum);
        }
        String token = json.hash(Map.of("mediaId", resource.id(), "version", resource.version(),
                "references", found, "copies", copies, "targets", deletionTargets(resource.id())));
        return new Preview(resource.id(), resource.version(), token,
                counts.entrySet().stream().map(entry -> new Usage(entry.getKey(), entry.getValue())).toList(),
                copies, blocked, deletions.selectById(resource.id()) != null);
    }

    private List<String> deletionTargets(String id) {
        var ids = new ArrayList<>(deletions.previewOutputs(id));
        ids.add(id);
        return ids.stream().distinct().sorted().toList();
    }

    private void authorize(MediaResource resource, CanvasActor actor) {
        access.editOwnership(resource, ResourceCommandContext.user(actor, null));
        if (resources.previewParent(resource.id()).isPresent()) {
            throw MediaException.invalid("预览文件由原素材管理，请通过原素材执行删除");
        }
    }

    private List<Reference> allReferences(String id) {
        return deletionTargets(id).stream().flatMap(target -> deletions.references(target).stream()).toList();
    }

    private void markDeleted(String targetId, String requestMediaId, CanvasActor actor) {
        var target = resources.get(targetId, true);
        if (target.resource().state().equals("DELETED")) {
            return;
        }
        media.update(Wrappers.<MediaResourceEntity>lambdaUpdate().eq(MediaResourceEntity::getId, targetId)
                .eq(MediaResourceEntity::getVersion, target.resource().version())
                .set(MediaResourceEntity::getState, "DELETED")
                .set(MediaResourceEntity::getVersion, target.resource().version() + 1)
                .set(MediaResourceEntity::getUpdatedAt, LocalDateTime.now(ZoneOffset.UTC)));
        tickets.update(Wrappers.<MediaAccessTicketEntity>lambdaUpdate()
                .eq(MediaAccessTicketEntity::getResourceId, targetId).set(MediaAccessTicketEntity::getRevoked, true));
        blobs.lockReady(target.blobId());
        boolean shared = media.exists(Wrappers.<MediaResourceEntity>lambdaQuery()
                .eq(MediaResourceEntity::getBlobId, target.blobId()).in(MediaResourceEntity::getState, "READY", "STAGING"));
        if (!shared) {
            blobs.update(Wrappers.<MediaBlobEntity>lambdaUpdate().eq(MediaBlobEntity::getId, target.blobId())
                    .set(MediaBlobEntity::getState, "ORPHAN"));
            if (purges.selectById(target.blobId()) == null) {
                var purge = new MediaPurgeEntity();
                purge.setBlobId(target.blobId());
                purge.setMediaId(requestMediaId);
                purges.insert(purge);
            }
        }
        if (!Objects.equals(targetId, requestMediaId)) {
            resources.event(resources.get(targetId, false).resource(), "media.deleted", actor.userId());
        }
    }
}
