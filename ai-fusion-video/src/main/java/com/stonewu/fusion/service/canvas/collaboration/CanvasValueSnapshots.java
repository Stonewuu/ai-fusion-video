package com.stonewu.fusion.service.canvas.collaboration;

import com.stonewu.fusion.mapper.canvas.CanvasValueSnapshotMapper;
import com.stonewu.fusion.entity.canvas.CanvasValueSnapshotEntity;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasOutputReference;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.ZoneOffset;
import java.util.Objects;
import java.util.UUID;

/**
 * 快照只由已授权复制事务创建；没有接受浏览器任意结果正文的写入入口。
 */
@Repository
@RequiredArgsConstructor
public class CanvasValueSnapshots {
    private final CanvasValueSnapshotMapper snapshots;
    private final CanvasJson json;

    public record Snapshot(String id, String canvasId, String nodeId, String portId, Value value,
                           String sourceCanvasId, String sourceNodeRunId, String sourceOutputId,
                           String sourceSnapshotId, Instant createdAt, String sourceKind, String templateHash) {
    }

    public Snapshot get(String id) {
        var row = snapshots.selectById(id);
        if (row == null) {
            throw new CanvasRunException(404, "FIXED_RESULT_NOT_FOUND", "固定结果快照不存在");
        }
        if (row.getSchemaVersion() != 1) {
            throw CanvasRunException.invalid("固定结果快照协议不受支持");
        }
        return new Snapshot(row.getId(), row.getCanvasId(), row.getNodeId(), row.getPortId(),
                json.read(row.getValueJson(), Value.class), row.getSourceCanvasId(), row.getSourceNodeRunId(),
                row.getSourceOutputId(), row.getSourceSnapshotId(), row.getCreatedAt().toInstant(ZoneOffset.UTC),
                row.getSourceKind(), row.getTemplateHash());
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public CanvasOutputReference copy(String sourceCanvas, CanvasOutputReference source, String targetCanvas,
                                      String targetNode, String port, Value value, CanvasActor actor) {
        if (!Objects.equals(value.id(), source.outputId())) {
            throw CanvasRunException.invalid("副本结果与原结果标识不一致");
        }
        String encoded = json.write(value);
        if (encoded.length() > 8 * 1024 * 1024) {
            throw CanvasRunException.invalid("固定结果超过 8 MiB，请缩小复制范围");
        }
        String id = UUID.randomUUID().toString();
        snapshots.create(
                CanvasValueSnapshotEntity.builder().id(id).canvasId(targetCanvas).nodeId(targetNode).portId(port)
                        .valueJson(encoded).sourceCanvasId(sourceCanvas).sourceNodeRunId(source.nodeRunId())
                        .sourceOutputId(source.outputId())
                        .sourceSnapshotId(source.snapshotId()).createdBy(actor.userId()).sourceKind("CANVAS").build());
        return new CanvasOutputReference(null, value.id(), value.mediaId(), id);
    }

    /**
     * 调用方必须先验证模板、创建新业务对象并完成实际媒体授权；不接受原库资源的伪造历史。
     */
    @Transactional(propagation = Propagation.MANDATORY)
    public CanvasOutputReference imported(String templateHash, String sourceOutput, String canvas, String node,
                                          String port, Value value, CanvasActor actor) {
        if (templateHash == null || !templateHash.matches("[0-9a-f]{64}")) {
            throw CanvasRunException.invalid("模板来源摘要无效");
        }
        String encoded = json.write(value);
        if (encoded.length() > 8 * 1024 * 1024) {
            throw CanvasRunException.invalid("固定结果超过 8 MiB，请缩小模板范围");
        }
        String id = UUID.randomUUID().toString();
        snapshots.create(CanvasValueSnapshotEntity.builder().id(id).canvasId(canvas).nodeId(node).portId(port)
                .valueJson(encoded).sourceOutputId(sourceOutput).createdBy(actor.userId())
                .sourceKind("TEMPLATE").templateHash(templateHash).build());
        return new CanvasOutputReference(null, value.id(), value.mediaId(), id);
    }
}
