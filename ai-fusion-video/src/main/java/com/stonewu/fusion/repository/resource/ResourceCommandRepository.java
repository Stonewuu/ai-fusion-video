package com.stonewu.fusion.repository.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.resource.ResourceCommandMapper;
import com.stonewu.fusion.entity.resource.ResourceCommandEntity;
import com.stonewu.fusion.entity.resource.ResourceEventEntity;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;
import java.util.Optional;

@Repository
@RequiredArgsConstructor
public class ResourceCommandRepository {
    private final ResourceCommandMapper commands;
    private final CanvasJson json;
    private final ResourceRevisionRepository revisions;

    public record Admission(String id, String hash, ResourceCommandReceipt receipt) {
    }

    public record PendingEvent(long id, String commandId, ResourceChange change) {
    }

    public Admission admit(String id, ResourceCommand command, ResourceCommandContext context, String hash) {
        commands.admit(ResourceCommandEntity.builder().id(id).actorUserId(context.actor().userId())
                .requestId(command.requestId()).requestHash(hash).origin(context.origin()).canvasId(context.canvasId())
                .agentRunId(context.agentRunId()).toolCallId(context.toolCallId())
                .requestJson(json.write(Map.of("schemaVersion", 1, "resourceType", command.resourceType(),
                        "operation", command.operation(), "input", command))).build());
        var row = commands.lockRequest(context.actor().userId(), command.requestId());
        return new Admission(row.getId(), row.getRequestHash(), row.getResultJson() == null ? null
                : json.read(row.getResultJson(), ResourceCommandReceipt.class));
    }

    public void commit(ResourceCommandReceipt receipt, ResourceMutation mutation) {
        int updated = commands.commit(json.write(receipt), json.write(mutation.previousValues()), receipt.commandId());
        if (updated != 1) {
            throw new IllegalStateException("业务命令重复提交");
        }
        revisions.capture(receipt, mutation);
        for (ResourceChange change : receipt.changes()) {
            commands.appendEvent(ResourceEventEntity.builder().eventId(change.eventId()).commandId(receipt.commandId())
                    .resourceType(change.resourceType().code()).resourceId(change.resourceId())
                    .projectId(change.projectId())
                    .resourceVersion(change.version()).deleted(change.deleted())
                    .changedFields(json.write(change.changedFields())).build());
        }
    }

    public Optional<PendingEvent> nextEvent() {
        // 自动增长 ID 不代表提交顺序；锁住流头直到当前事件和投影一同提交。
        commands.head(true);
        return Optional.ofNullable(commands.nextEvent()).map(row -> new PendingEvent(row.getId(), row.getCommandId(),
                new ResourceChange(row.getEventId(), CanvasValueType.fromCode(row.getResourceType()),
                        row.getResourceId(),
                        row.getProjectId(), row.getResourceVersion(), row.getDeleted(),
                        List.of(json.read(row.getChangedFields(), String[].class)))));
    }

    public void markPublished(long eventId) {
        long sequence = commands.head(true) + 1;
        commands.appendFeed(sequence, eventId);
        commands.advanceHead(sequence);
        if (commands.markPublished(eventId) != 1) {
            throw new IllegalStateException("资源事件投递状态已变化");
        }
    }

    public List<String> targetCanvases(ResourceChange change) {
        return commands.targetCanvases(change.projectId(), change.resourceType().code(), change.resourceId());
    }

    public void project(PendingEvent event, String canvasId, long sequence) {
        ObjectNode payload = json.object(event.change());
        payload.put("resourceCommandId", event.commandId());
        commands.projectEvent(canvasId, sequence, json.write(payload));
        commands.recordProjection(event.change().eventId(), canvasId, sequence);
    }
}
