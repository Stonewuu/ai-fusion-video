package com.stonewu.fusion.repository.canvas;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.stonewu.fusion.entity.canvas.CanvasMemberEntity;

import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.canvas.CanvasMapper;
import com.stonewu.fusion.mapper.canvas.CanvasMemberMapper;
import com.stonewu.fusion.entity.canvas.CanvasEntity;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.service.canvas.model.*;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;
import java.util.Optional;

@Repository
@RequiredArgsConstructor
public class CanvasDocumentRepository {
    private final CanvasMapper documents;
    private final CanvasMemberMapper members;

    public Optional<CanvasDocument> find(String id, boolean lock) {
        return Optional.ofNullable(lock ? documents.lock(id) : documents.selectById(id)).map(this::map);
    }

    public CanvasDocument require(String id, boolean lock) {
        return find(id, lock).orElseThrow(() -> new CanvasException(404, CanvasErrorCode.RESOURCE_DELETED,
                "画布不存在", Map.of()));
    }

    public void admit(String id, String title, Long projectId, int ownerType, long ownerId, long userId) {
        documents.admit(id, title, projectId, ownerType, ownerId, userId);
    }

    public void advance(CanvasDocument document) {
        if (document.revision() == Long.MAX_VALUE || document.nextEventSequence() == Long.MAX_VALUE) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "画布版本已达到上限");
        }
        int updated = documents.advance(document.id(), document.revision(), document.nextEventSequence());
        if (updated != 1) {
            throw new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT, "画布版本已变化", Map.of());
        }
    }

    public boolean isProjectManager(long projectId, long userId) {
        return documents.countProjectManagers(projectId, userId) > 0;
    }

    public Optional<CanvasPermissions> memberPermissions(String canvasId, long userId) {
        return Optional.ofNullable(members.selectOne(Wrappers.<CanvasMemberEntity>lambdaQuery()
                        .eq(CanvasMemberEntity::getCanvasId, canvasId).eq(CanvasMemberEntity::getUserId, userId)))
                .map(row -> new CanvasPermissions(true, row.getCanEdit(), row.getCanExecute(), row.getCanManage()));
    }

    public PageResult<CanvasListItem> list(long userId, List<Long> projects, Long projectId, String search,
                                           boolean archived, int pageNo, int pageSize) {
        String pattern = search.isEmpty() ? null : "%" + search.replace("!", "!!").replace("%", "!%")
                .replace("_", "!_") + "%";
        var page = documents.accessible(new Page<>(pageNo, pageSize), userId, projects, projectId, pattern, archived);
        return new PageResult<>(
                page.getRecords().stream().map(row -> new CanvasListItem(map(row), row.getProjectName())).toList(),
                page.getTotal());
    }

    private CanvasDocument map(CanvasEntity row) {
        return new CanvasDocument(row.getId(), row.getTitle(), row.getProjectId(),
                row.getOwnerType(), row.getOwnerId(), row.getCreatedBy(),
                row.getSchemaVersion(),
                row.getRevision(), row.getNextEventSequence(), row.getArchived(),
                row.getCreateTime(), row.getUpdateTime());
    }
}
