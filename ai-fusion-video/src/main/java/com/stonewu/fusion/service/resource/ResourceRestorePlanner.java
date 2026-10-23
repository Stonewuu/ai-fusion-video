package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.repository.resource.ResourceRevisionRepository;
import com.stonewu.fusion.repository.resource.ResourceRevisionRepository.Revision;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.media.MediaRestoreValidation;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;

@Component
@RequiredArgsConstructor
public class ResourceRestorePlanner {
    private static final Set<String> IMMUTABLE = Set.of("id", "ownerType", "ownerId", "scope", "projectId", "scriptId",
            "storyboardId", "assetId", "createTime", "updateTime", "version");
    private final CanvasDocumentRepository canvases;
    private final CanvasAccessPolicy access;
    private final ResourceHistoryService history;
    private final ResourceRevisionRepository revisions;
    private final ResourceReadRepository resources;
    private final ResourceRestoreRelations relations;
    private final MediaRestoreValidation media;
    private final CanvasJson json;

    public record Change(CanvasValueType resourceType, long resourceId, Long projectId, int currentVersion,
                         ObjectNode current, ObjectNode proposed, List<String> fields, boolean removesCreatedObject) {
    }

    public record Conflict(CanvasValueType resourceType, long resourceId, String field, String message) {
    }

    public record Plan(String commandId, CanvasValueType resourceType, String digest, List<Change> changes,
                       List<Conflict> conflicts, List<String> notices) {
        @JsonProperty("canRestore")
        public boolean canRestore() {
            return conflicts.isEmpty() && !changes.isEmpty();
        }
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Plan preview(String canvasId, String commandId, CanvasActor actor) {
        return prepare(canvasId, commandId, actor, false);
    }

    public CanvasDocument authorize(String canvasId, String commandId, CanvasActor actor) {
        var document = canvases.require(canvasId, false);
        var source = history.source(commandId);
        history.authorize(document, source, actor, true);
        if (type(source) == CanvasValueType.PROJECT) {
            access.requireProjectManager(document.projectId(), actor);
        }
        return document;
    }

    public Plan prepare(String canvasId, String commandId, CanvasActor actor, boolean lock) {
        var document = authorize(canvasId, commandId, actor);
        var source = history.source(commandId);
        var entries = revisions.command(commandId).stream()
                .sorted(Comparator.comparingInt((Revision entry) -> order(entry.resourceType()))
                        .thenComparingLong(Revision::resourceId)).toList();
        if (entries.isEmpty()) {
            throw new CanvasException(CanvasErrorCode.UNSUPPORTED_SCHEMA,
                    "这项旧操作没有完整前后版本，不能自动恢复；原始审计仍可查看");
        }
        if (lock && document.projectId() != null) {
            resources.raw(CanvasValueType.PROJECT, document.projectId(), true);
        }
        boolean structural = entries.stream().anyMatch(
                entry -> entry.before() == null || entry.fields().contains("deleted") || entry.fields()
                        .contains("sortOrder"));
        var changes = new ArrayList<Change>();
        var conflicts = new ArrayList<Conflict>();
        var notices = new ArrayList<String>();
        for (var entry : entries) {
            ObjectNode current = resources.raw(entry.resourceType(), entry.resourceId(), lock);
            int version = current.path("version").intValue();
            if (entry.resourceType() == CanvasValueType.PROJECT && (entry.before() == null || entry.fields()
                    .contains("deleted"))) {
                conflicts.add(new Conflict(entry.resourceType(), entry.resourceId(), "deleted",
                        "项目创建和删除涉及整个工作区及成员权限，不能通过画布内容历史恢复"));
            }
            if (version == Integer.MAX_VALUE) {
                throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "业务版本已达到上限");
            }
            if (structural && version != entry.version()) {
                conflicts.add(
                        new Conflict(entry.resourceType(), entry.resourceId(), "version",
                                "结构或新建对象之后已被修改，不能覆盖后续工作"));
            }
            ObjectNode proposed = current.deepCopy();
            if (entry.before() == null) {
                proposed.put("deleted", true);
                if (proposed.has("deletedId")) {
                    proposed.put("deletedId", entry.resourceId());
                }
            } else {
                for (String field : entry.fields()) {
                    if (IMMUTABLE.contains(field)) {
                        throw new IllegalStateException("历史命令包含不可恢复的归属字段：" + field);
                    }
                    if (!current.path(field).equals(entry.after().path(field))) {
                        conflicts.add(
                                new Conflict(entry.resourceType(), entry.resourceId(), field,
                                        "此字段在原操作之后又有修改"));
                    }
                    proposed.set(field, entry.before().path(field));
                }
            }
            if (current.path("deleted").asBoolean() && !entry.after().path("deleted").asBoolean()) {
                conflicts.add(
                        new Conflict(entry.resourceType(), entry.resourceId(), "deleted", "对象已被后续操作删除"));
            }
            proposed.put("version", version + 1);
            String runtimeField = switch (entry.resourceType()) {
                case SCRIPT, SCRIPT_EPISODE -> "parsingStatus";
                case STORYBOARD_EPISODE -> "composeStatus";
                default -> null;
            };
            if (runtimeField != null && !proposed.path("deleted").asBoolean() && proposed.path(runtimeField)
                    .asInt() == 1
                    && (entry.fields().contains(runtimeField) || current.path("deleted").asBoolean())) {
                if (runtimeField.equals("parsingStatus")) {
                    proposed.put("parsingStatus", 0);
                    if (proposed.has("parsingProgress")) {
                        proposed.putNull("parsingProgress");
                    }
                    notices.add("旧解析任务不会重新启动，恢复后的解析状态为待处理。");
                } else {
                    boolean hasVideo = !proposed.path("composedVideoUrl").asText("").isBlank();
                    proposed.put("composeStatus", hasVideo ? 2 : 0);
                    proposed.putNull("composeJobId");
                    proposed.putNull("composeErrorMsg");
                    if (!hasVideo) {
                        proposed.putNull("composedAt");
                    }
                    notices.add("旧合成任务不会重新启动，已有成片会保留，执行中状态会清除。");
                }
            }
            changes.add(
                    new Change(entry.resourceType(), entry.resourceId(), entry.projectId(), version, current, proposed,
                            ResourceRevisionRepository.changedFields(current, proposed), entry.before() == null));
        }
        if (conflicts.isEmpty()) {
            relations.validate(changes, actor);
            media.validate(changes, actor, lock);
        }
        String digest = json.hash(
                Map.of("sourceCommandId", commandId, "canvasId", canvasId, "actor", actor, "changes", changes));
        return new Plan(commandId, type(source), digest, List.copyOf(changes), List.copyOf(conflicts),
                notices.stream().distinct().toList());
    }

    private CanvasValueType type(ResourceHistoryService.Source source) {
        if (source.request().path("schemaVersion").asInt() != 1) {
            throw new CanvasException(CanvasErrorCode.UNSUPPORTED_SCHEMA,
                    "这项历史操作的恢复协议不受支持，原始审计仍可查看");
        }
        return CanvasValueType.fromCode(source.request().path("resourceType").asText());
    }

    public static int order(CanvasValueType type) {
        return switch (type) {
            case PROJECT -> 0;
            case SCRIPT -> 1;
            case ASSET -> 2;
            case ASSET_ITEM -> 3;
            case SCRIPT_EPISODE -> 4;
            case SCRIPT_SCENE -> 5;
            case STORYBOARD -> 6;
            case STORYBOARD_EPISODE -> 7;
            case STORYBOARD_SCENE -> 8;
            case SHOT -> 9;
            default -> throw new IllegalArgumentException("不是业务类型");
        };
    }
}
