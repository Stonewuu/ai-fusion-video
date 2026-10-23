package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.repository.resource.ResourceRevisionRepository;
import com.stonewu.fusion.repository.resource.ResourceRevisionRepository.Revision;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasRequestValidator;
import com.stonewu.fusion.service.canvas.collaboration.CanvasCollaborationService;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.resource.model.ResourceChange;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.canvas.CanvasNodeEntity;
import com.stonewu.fusion.mapper.resource.ResourceCommandMapper;
import com.stonewu.fusion.mapper.canvas.CanvasNodeMapper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.*;

@Service
@RequiredArgsConstructor
public class ResourceHistoryService {
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy canvases;
    private final ResourceReadRepository resources;
    private final ResourceAccessPolicy access;
    private final ResourceRevisionRepository revisions;
    private final CanvasCollaborationService people;
    private final CanvasRequestValidator validation;
    private final CanvasJson json;
    private final ResourceHistoryCache cache;
    private final ResourceCommandMapper commands;
    private final CanvasNodeMapper nodes;

    public record Source(String id, long userId, String origin, String canvasId, String agentRunId, String toolCallId,
                         String operation, ResourceCommandReceipt receipt, ObjectNode request, ObjectNode previous,
                         LocalDateTime committedAt) {
    }

    public record Summary(String commandId, long actorUserId, String actorName, String origin, String operation,
                          List<ResourceChange> resources, Instant committedAt, int snapshotSchemaVersion) {
    }

    public record Page(List<Summary> items, String nextCursor) {
    }

    public record Detail(Summary summary, List<Revision> revisions, ObjectNode originalRequest,
                         ObjectNode originalPrevious,
                         String agentRunId, String toolCallId) {
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Page list(String canvasId, String before, int limit, CanvasActor actor) {
        var document = readable(canvasId, actor);
        if (limit < 1 || limit > 100) {
            throw invalid("历史记录每页最多 100 项");
        }
        LocalDateTime beforeTime = null;
        String beforeId = null;
        if (before != null && !before.isBlank()) {
            String decoded;
            try {
                decoded = new String(Base64.getUrlDecoder().decode(before), StandardCharsets.UTF_8);
            } catch (IllegalArgumentException invalid) {
                throw invalid("历史游标无效");
            }
            var parts = decoded.split("\\|", -1);
            if (parts.length != 2) {
                throw invalid("历史游标无效");
            }
            validation.id(parts[1]);
            try {
                beforeTime = LocalDateTime.parse(parts[0]);
            } catch (RuntimeException invalid) {
                throw invalid("历史游标时间无效");
            }
            beforeId = parts[1];
        }
        var ids = commands.history(document.id(), document.projectId(), beforeTime, beforeId, limit + 1);
        var items = new ArrayList<Summary>();
        Source last = null;
        for (String id : ids.stream().limit(limit).toList()) {
            Source source = source(id);
            last = source;
            try {
                authorize(document, source, actor, false);
            } catch (CanvasException denied) {
                if (denied.getCode() != 403 && denied.getCode() != 404) {
                    throw denied;
                }
                continue;
            }
            items.add(summary(source));
        }
        String next = ids.size() > limit && last != null ? Base64.getUrlEncoder().withoutPadding().encodeToString(
                (last.committedAt() + "|" + last.id()).getBytes(StandardCharsets.UTF_8)) : null;
        return new Page(List.copyOf(items), next);
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Detail detail(String canvasId, String commandId, CanvasActor actor) {
        var document = readable(canvasId, actor);
        var source = source(commandId);
        authorize(document, source, actor, false);
        return new Detail(summary(source), revisions.command(commandId), source.request(), source.previous(),
                source.agentRunId(), source.toolCallId());
    }

    public Source source(String commandId) {
        validation.id(commandId);
        return json.read(cache.source(commandId), Source.class);
    }

    public void authorize(CanvasDocument document, Source source, CanvasActor actor, boolean write) {
        if (write) {
            canvases.requireEdit(document, actor);
        } else {
            canvases.requireView(document, actor);
        }
        if (source.receipt().changes().isEmpty()) {
            if (!Objects.equals(source.canvasId(), document.id())) {
                throw denied();
            }
            return;
        }
        for (var change : source.receipt().changes()) {
            if (!Objects.equals(change.projectId(), document.projectId())) {
                throw denied();
            }
            if (document.projectId() == null && !Objects.equals(source.canvasId(), document.id())) {
                long references = nodes.selectCount(new LambdaQueryWrapper<CanvasNodeEntity>()
                        .eq(CanvasNodeEntity::getCanvasId, document.id())
                        .eq(CanvasNodeEntity::getResourceType, change.resourceType().code())
                        .eq(CanvasNodeEntity::getResourceId, change.resourceId()));
                if (references == 0) {
                    throw denied();
                }
            }
            var current = resources.raw(change.resourceType(), change.resourceId(), false);
            var header = new ResourceHeader(change.resourceType(), change.resourceId(), change.projectId(),
                    current.path("version").intValue(), current.path("deleted").asBoolean());
            if (write) {
                access.requireEdit(header, actor);
            } else {
                access.require(header, actor);
            }
        }
    }

    private Summary summary(Source source) {
        return new Summary(source.id(), source.userId(), people.person(source.userId()).name(), source.origin(),
                source.operation(),
                source.receipt().changes(), source.committedAt().toInstant(ZoneOffset.UTC),
                source.request().path("schemaVersion").asInt() == 1 && !revisions.command(source.id())
                        .isEmpty() ? 1 : 0);
    }


    private CanvasDocument readable(String canvasId, CanvasActor actor) {
        validation.id(canvasId);
        var document = documents.require(canvasId, false);
        canvases.requireView(document, actor);
        return document;
    }

    private CanvasException invalid(String message) {
        return new CanvasException(CanvasErrorCode.INVALID_COMMAND, message);
    }

    private CanvasException denied() {
        return new CanvasException(403, CanvasErrorCode.ACCESS_DENIED, "无权查看或恢复这项业务操作", Map.of());
    }
}
