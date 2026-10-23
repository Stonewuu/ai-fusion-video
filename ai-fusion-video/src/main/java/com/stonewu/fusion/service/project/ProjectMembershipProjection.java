package com.stonewu.fusion.service.project;

import com.stonewu.fusion.mapper.project.ProjectMembershipCommandMapper;
import com.stonewu.fusion.entity.project.ProjectMembershipCommandEntity;
import com.stonewu.fusion.mapper.canvas.CanvasMapper;
import com.stonewu.fusion.entity.canvas.CanvasEntity;
import com.stonewu.fusion.mapper.canvas.CanvasEventMapper;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasSnapshotCache;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import lombok.extern.slf4j.Slf4j;

import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 权限事务只锁项目；通知在提交后从持久化发件箱投递，避免项目/画布锁顺序倒置。
 */
@Component
@Slf4j
public class ProjectMembershipProjection {
    private final ProjectMembershipCommandMapper commands;
    private final CanvasMapper canvasRows;
    private final CanvasEventMapper events;
    private final CanvasDocumentRepository canvases;
    private final CanvasSnapshotCache snapshots;
    private final CanvasJson json;
    private final TransactionTemplate writes;

    public ProjectMembershipProjection(ProjectMembershipCommandMapper commands, CanvasMapper canvasRows,
                                       CanvasEventMapper events, CanvasDocumentRepository canvases,
                                       CanvasSnapshotCache snapshots, CanvasJson json,
                                       PlatformTransactionManager manager) {
        this.commands = commands;
        this.canvasRows = canvasRows;
        this.events = events;
        this.canvases = canvases;
        this.snapshots = snapshots;
        this.json = json;
        writes = new TransactionTemplate(manager);
    }


    private record Published(boolean found, List<CanvasDocument> changed) {
    }

    public boolean publishNext() {
        Published result = Objects.requireNonNull(writes.execute(ignored -> {
            var pending = commands.nextUnpublished();
            if (pending == null) {
                return new Published(false, List.of());
            }
            var event = pending;
            List<CanvasDocument> changed = new ArrayList<>();
            for (String id : canvasRows.selectList(Wrappers.<CanvasEntity>lambdaQuery().select(CanvasEntity::getId)
                            .eq(CanvasEntity::getProjectId, event.getProjectId()).orderByAsc(CanvasEntity::getId)).stream()
                    .map(CanvasEntity::getId).toList()) {
                var canvas = canvases.require(id, true);
                events.membershipChanged(id, canvas.nextEventSequence(),
                        json.write(Map.of("projectId", event.getProjectId(),
                                "membershipRevision", event.getRevision(), "membershipCommandId", event.getId())));
                canvases.advance(canvas);
                changed.add(canvas);
            }
            commands.update(Wrappers.<ProjectMembershipCommandEntity>lambdaUpdate()
                    .eq(ProjectMembershipCommandEntity::getId, event.getId())
                    .set(ProjectMembershipCommandEntity::getPublished, true));
            return new Published(true, changed);
        }));
        result.changed().forEach(snapshots::evict);
        return result.found();
    }

    @Scheduled(fixedDelayString = "${app.canvas.membership-poll-ms:500}", initialDelay = 2000)
    public void publish() {
        try {
            for (int i = 0; i < 50 && publishNext(); i++) {
            }
        } catch (RuntimeException failure) {
            log.error("项目成员通知投递失败，将从持久化记录继续", failure);
        }
    }
}
