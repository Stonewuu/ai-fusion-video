package com.stonewu.fusion.repository.canvas;

import com.stonewu.fusion.mapper.canvas.CanvasCommentMapper;
import com.stonewu.fusion.mapper.canvas.CanvasRevisionMapper;
import com.stonewu.fusion.mapper.canvas.CanvasMapper;
import com.stonewu.fusion.entity.canvas.CanvasCommentEntity;
import com.stonewu.fusion.entity.canvas.CanvasRevisionEntity;
import com.stonewu.fusion.entity.canvas.CanvasEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import java.util.Optional;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.resource.model.ResourceSnapshot;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Repository;

import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;

@Repository
@RequiredArgsConstructor
public class CanvasWorkspaceRepository {
    private final CanvasCommentMapper comments;
    private final CanvasRevisionMapper revisions;
    private final CanvasMapper documents;
    private final CanvasJson json;

    public record Comment(String id, String nodeId, String parentId, Long resourceVersion, String content,
                          long createdBy, boolean resolved, long version, boolean deleted, Instant createdAt,
                          Instant updatedAt, String authorName) {
    }

    public record Checkpoint(String id, String title, long revision, long sequence, long createdBy, Instant createdAt,
                             String authorName) {
    }

    public record CheckpointContent(CanvasGraph graph, List<ResourceSnapshot> resources) {
    }

    public record CheckpointDetail(Checkpoint checkpoint, CheckpointContent snapshot) {
    }

    public List<Comment> comments(String canvasId, int offset, int limit) {
        return comments.history(canvasId, limit, offset).stream().map(this::comment).toList();
    }

    public Comment comment(String canvasId, String id) {
        return Optional.ofNullable(comments.detail(canvasId, id)).map(this::comment).orElseThrow(() -> missing("评论"));
    }

    public void createComment(String canvasId, String id, String nodeId, String parentId, Long resourceVersion,
                              String content, long actor) {
        comments.create(id, canvasId, nodeId, parentId, resourceVersion, content, actor);
    }

    public void patchComment(String canvasId, Comment before, String content, boolean resolved, boolean deleted) {
        int updated = comments.patch(content, resolved, deleted, canvasId, before.id(), before.version());
        if (updated != 1) {
            throw new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT, "评论已被修改或删除", Map.of());
        }
    }

    public void patchDocument(String canvasId, String title, boolean archived) {
        documents.update(Wrappers.<CanvasEntity>lambdaUpdate().eq(CanvasEntity::getId, canvasId)
                .set(CanvasEntity::getTitle, title).set(CanvasEntity::getArchived, archived));
    }

    public void createCheckpoint(String canvasId, String id, String title, long revision, long sequence,
                                 long actor, CheckpointContent snapshot) {
        revisions.create(id, canvasId, title, revision, sequence, json.write(snapshot), actor);
    }

    public List<Checkpoint> checkpoints(String canvasId, long beforeRevision, int limit) {
        return revisions.history(canvasId, beforeRevision, limit).stream().map(this::checkpoint).toList();
    }

    public CheckpointDetail checkpoint(String canvasId, String id) {
        return Optional.ofNullable(revisions.detail(canvasId, id))
                .map(row -> new CheckpointDetail(checkpoint(row),
                        json.read(row.getSnapshotJson(), CheckpointContent.class)))
                .orElseThrow(() -> missing("命名版本"));
    }

    private Comment comment(CanvasCommentEntity row) {
        boolean deleted = row.getDeleted();
        return new Comment(row.getId(), row.getNodeId(), row.getParentId(),
                row.getResourceVersion(),
                deleted ? "" : row.getContent(), row.getCreatedBy(), row.getResolved(),
                row.getVersion(), deleted,
                row.getCreateTime().toInstant(ZoneOffset.UTC), row.getUpdateTime().toInstant(ZoneOffset.UTC),
                row.getAuthorName());
    }

    private Checkpoint checkpoint(CanvasRevisionEntity row) {
        return new Checkpoint(row.getId(), row.getTitle(), row.getRevision(),
                row.getSequenceNo(),
                row.getCreatedBy(), row.getCreateTime().toInstant(ZoneOffset.UTC), row.getAuthorName());
    }


    private CanvasException missing(String kind) {
        return new CanvasException(404, CanvasErrorCode.RESOURCE_DELETED, kind + "不存在", Map.of());
    }
}
