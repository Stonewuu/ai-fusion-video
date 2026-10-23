package com.stonewu.fusion.repository.canvas;

import com.stonewu.fusion.mapper.canvas.CanvasOperationMapper;
import com.stonewu.fusion.mapper.canvas.CanvasEventMapper;
import com.stonewu.fusion.entity.canvas.CanvasOperationEntity;
import com.stonewu.fusion.entity.canvas.CanvasEventEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import com.stonewu.fusion.service.canvas.command.CanvasGraphChangeSet;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasEvent;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Repository;

import java.time.Instant;
import java.time.ZoneOffset;
import java.util.*;

@Repository
@RequiredArgsConstructor
public class CanvasJournalRepository {
    private final CanvasOperationMapper operations;
    private final CanvasEventMapper eventRows;
    private final CanvasJson json;

    public record StoredCommand(String requestHash, CanvasCommandReceipt receipt) {
    }

    public record UndoSource(CanvasCommandReceipt receipt, CanvasGraphChangeSet before) {
    }

    public record HistoryEntry(String commandId, long revision, long actorUserId, String origin, String eventType,
                               List<String> nodeIds, int edgeCount, boolean canUndo, Instant createdAt,
                               String actorName) {
    }

    public UndoSource undoSource(String canvasId, String commandId) {
        var row = operations.undoSource(canvasId, commandId);
        if (row == null) {
            throw new CanvasException(404, CanvasErrorCode.RESOURCE_DELETED, "可撤销的图命令不存在", Map.of());
        }
        return new UndoSource(json.read(row.getResultJson(), CanvasCommandReceipt.class),
                json.read(row.getUndoJson(), CanvasGraphChangeSet.class));
    }

    public List<HistoryEntry> history(String canvasId, long beforeRevision, int limit) {
        return operations.history(canvasId, beforeRevision, limit).stream().map(row -> {
            var changes = json.read(row.getResultJson(), CanvasCommandReceipt.class).changes();
            var ids = new ArrayList<>(changes.deletedNodeIds());
            changes.upsertedNodes().forEach(node -> ids.add(node.id()));
            int edges = changes.upsertedEdges().size() + changes.deletedEdgeIds().size();
            String type = row.getEventType();
            return new HistoryEntry(row.getId(), row.getCommittedRevision(), row.getActorUserId(), row.getOrigin(),
                    type, List.copyOf(ids), edges,
                    (type.equals("canvas.graph.changed") || type.equals("canvas.graph.undone") || type.equals(
                            "canvas.checkpoint.restored"))
                            && (!ids.isEmpty() || edges > 0),
                    row.getCreateTime().toInstant(ZoneOffset.UTC), row.getActorName());
        }).toList();
    }

    public Optional<StoredCommand> find(String canvasId, long userId, String requestId) {
        return Optional.ofNullable(operations.selectOne(Wrappers.<CanvasOperationEntity>lambdaQuery()
                        .eq(CanvasOperationEntity::getCanvasId, canvasId).eq(CanvasOperationEntity::getActorUserId, userId)
                        .eq(CanvasOperationEntity::getRequestId, requestId)))
                .map(row -> new StoredCommand(row.getRequestHash(),
                        json.read(row.getResultJson(), CanvasCommandReceipt.class)));
    }

    public void append(long userId, String origin, String agentRunId, String toolCallId,
                       String requestHash, Object request, CanvasCommandReceipt receipt,
                       CanvasGraphChangeSet previousValues, String eventType) {
        operations.append(CanvasOperationEntity.builder()
                .id(receipt.commandId())
                .canvasId(receipt.canvasId())
                .actorUserId(userId)
                .requestId(receipt.requestId())
                .requestHash(requestHash)
                .origin(origin)
                .agentRunId(agentRunId)
                .toolCallId(toolCallId)
                .committedRevision(receipt.revision())
                .requestJson(json.write(request))
                .resultJson(json.write(receipt))
                .undoJson(json.write(previousValues)).build());
        eventRows.append(CanvasEventEntity.builder()
                .id(UUID.randomUUID().toString())
                .canvasId(receipt.canvasId())
                .sequenceNo(receipt.sequence())
                .operationId(receipt.commandId())
                .eventType(eventType)
                .payloadJson(json.write(receipt)).build());
    }

    public List<CanvasEvent> events(String canvasId, long afterSequence, int limit) {
        return eventRows.selectPage(new Page<CanvasEventEntity>(1, limit, false),
                        Wrappers.<CanvasEventEntity>lambdaQuery()
                                .eq(CanvasEventEntity::getCanvasId, canvasId)
                                .gt(CanvasEventEntity::getSequenceNo, afterSequence)
                                .orderByAsc(CanvasEventEntity::getSequenceNo)).getRecords().stream()
                .map(row -> new CanvasEvent(row.getId(), row.getCanvasId(), row.getSequenceNo(), row.getOperationId(),
                        row.getEventType(), json.read(row.getPayloadJson(), ObjectNode.class), row.getCreateTime()))
                .toList();
    }

    public Long firstSequence(String canvasId) {
        return eventRows.firstSequence(canvasId);
    }
}
