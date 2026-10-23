package com.stonewu.fusion.service.canvas.delivery;

import com.stonewu.fusion.mapper.canvas.CanvasExportMapper;
import com.stonewu.fusion.mapper.canvas.CanvasExportBlobMapper;
import com.stonewu.fusion.mapper.storage.StorageConfigMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.entity.canvas.CanvasExportEntity;
import com.stonewu.fusion.entity.canvas.CanvasExportBlobEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import java.time.LocalDateTime;
import java.util.Optional;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.MediaCryptor;
import com.stonewu.fusion.service.media.model.MediaBlob;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.storage.StorageTypes;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Repository;

import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static com.stonewu.fusion.service.canvas.delivery.CanvasExportData.*;

@Repository
@RequiredArgsConstructor
public class CanvasExportRepository {
    private final CanvasExportMapper exports;
    private final CanvasExportBlobMapper blobs;
    private final StorageConfigMapper configurations;
    private final UserMapper users;
    private final CanvasJson json;
    private final MediaCryptor cryptor;

    public Job get(String id, boolean lock) {
        return Optional.ofNullable(lock ? exports.lock(id) : exports.selectById(id)).map(this::map)
                .orElseThrow(() -> new BusinessException(404, "交付记录不存在"));
    }

    public Job replay(String canvas, CanvasActor actor, String request, String hash) {
        var row = exports.selectOne(
                Wrappers.<CanvasExportEntity>lambdaQuery().eq(CanvasExportEntity::getCanvasId, canvas)
                        .eq(CanvasExportEntity::getActorUserId, actor.userId())
                        .eq(CanvasExportEntity::getActorTeamId, team(actor))
                        .eq(CanvasExportEntity::getRequestId, request));
        if (row == null) {
            return null;
        }
        if (!hash.equals(row.getRequestHash())) {
            throw conflict("此请求编号已用于不同的导出内容");
        }
        return map(row);
    }

    public Job insert(String canvas, String nodeRunId, String request, String hash, Format format, Manifest manifest,
                      ResourceCommandContext context) {
        if (exports.selectCount(Wrappers.<CanvasExportEntity>lambdaQuery()
                .eq(CanvasExportEntity::getActorUserId, context.actor().userId())
                .in(CanvasExportEntity::getStatus, List.of("QUEUED", "RUNNING"))) >= 8) {
            throw conflict("最多同时生成 8 个交付文件，请等待或取消现有任务");
        }
        String id = UUID.randomUUID().toString();
        String filename = CanvasExportWriter.filename(manifest, format);
        exports.create(CanvasExportEntity.builder()
                .id(id)
                .canvasId(canvas)
                .nodeRunId(nodeRunId)
                .actorUserId(context.actor().userId())
                .actorTeamId(team(context.actor()))
                .requestId(request)
                .requestHash(hash)
                .format(format.name())
                .contextJson(json.write(context))
                .manifestJson(json.write(manifest))
                .manifestHash(json.hash(manifest))
                .filename(filename)
                .mimeType(CanvasExportWriter.mime(manifest, format)).build());
        event(id, context.actor().userId(), "CREATED", Map.of("format", format, "manifestHash", json.hash(manifest)));
        return get(id, false);
    }

    public List<Job> list(String canvas, int page) {
        return exports.selectPage(new Page<CanvasExportEntity>(page + 1, 40, false),
                        Wrappers.<CanvasExportEntity>lambdaQuery()
                                .eq(CanvasExportEntity::getCanvasId, canvas)
                                .orderByDesc(CanvasExportEntity::getCreateTime, CanvasExportEntity::getId))
                .getRecords().stream().map(this::map).toList();
    }

    public List<String> pending() {
        return exports.pending();
    }

    public Claim claim(String id) {
        Job job = get(id, true);
        if (job.terminal() || ("RUNNING".equals(job.status()) && job.leaseUntil() != null && job.leaseUntil()
                .isAfter(Instant.now()))) {
            return null;
        }
        if (job.cancelRequested()) {
            finishState(job, "CANCELLED", null);
            return null;
        }
        if (job.attempts() >= 3) {
            finishState(job, "FAILED", "多次重启未能完成交付，请重新创建导出任务");
            return null;
        }
        orphan(id);
        String worker = UUID.randomUUID().toString();
        exports.claim(worker, id);
        event(id, null, "CLAIMED", Map.of("attempt", job.attempts() + 1, "epoch", job.epoch() + 1));
        return new Claim(id, worker, job.epoch() + 1);
    }

    public Job leased(Claim claim, boolean lock) {
        var job = get(claim.id(), lock);
        if (!"RUNNING".equals(job.status()) || !claim.worker().equals(job.worker()) || claim.epoch() != job.epoch()
                || job.leaseUntil() == null || !job.leaseUntil().isAfter(Instant.now())) {
            throw conflict("交付工作者租约已经失效");
        }
        return job;
    }

    public void heartbeat(Claim claim) {
        int changed = exports.heartbeat(claim.id(), claim.worker(), claim.epoch());
        if (changed != 1) {
            throw conflict("交付任务已取消或工作者租约失效");
        }
    }

    public MediaBlob reserve(Job job) {
        var configs = configurations.lockActiveDefaults();
        if (configs.size() != 1) {
            throw conflict("需要唯一且可用的默认存储配置来保存交付文件");
        }
        var config = configs.getFirst();
        String backend = StorageTypes.normalizeType(config.getType());
        if (!List.of("local", "s3").contains(backend)) {
            throw invalid("交付文件需要本地或 S3 存储");
        }
        if (backend.equals("local") && (config.getBasePath() == null || config.getBasePath().toString()
                .isBlank())) {
            throw invalid("默认本地存储需要配置明确的根目录");
        }
        var blob = new MediaBlob(UUID.randomUUID().toString(), job.id(), config.getId(),
                backend, UUID.randomUUID() + ".afvblob", cryptor.newKeyset(), null);
        blobs.reserve(blob.id(), job.id(), blob.storageConfigId(), blob.backend(), blob.objectKey(), blob.keyset(),
                job.epoch());
        return blob;
    }

    public MediaBlob blob(String id) {
        return Optional.ofNullable(blobs.selectOne(Wrappers.<CanvasExportBlobEntity>lambdaQuery()
                        .eq(CanvasExportBlobEntity::getId, id).eq(CanvasExportBlobEntity::getState, "READY")))
                .map(CanvasExportRepository::blob).orElseThrow(() -> new BusinessException(404, "交付文件尚未就绪"));
    }

    public void complete(Job job, MediaBlob blob, long storedBytes, long size, String hash) {
        if (!job.id().equals(blob.resourceId())) {
            throw new IllegalStateException("交付文件归属不一致");
        }
        int written = blobs.complete(storedBytes, blob.id(), job.id(), job.epoch());
        if (written != 1) {
            throw conflict("交付文件登记已经失效");
        }
        exports.complete(blob.id(), size, hash, job.id());
        event(job.id(), null, "SUCCEEDED", Map.of("byteSize", size, "contentHash", hash));
    }

    public void cancel(Job job, long actor) {
        if (job.terminal() || job.cancelRequested()) {
            return;
        }
        exports.cancel(job.id());
        event(job.id(), actor, "CANCEL_REQUESTED", Map.of());
        if (job.status().equals("QUEUED")) {
            finishState(job, "CANCELLED", null);
        }
    }

    public void finishState(Job job, String state, String error) {
        orphan(job.id());
        exports.finish(state, error, job.id());
        event(job.id(), null, state, error == null ? Map.of() : Map.of("message", error));
    }

    public void event(String id, Long actor, String type, Object detail) {
        exports.event(actor, type, json.write(detail), id);
    }

    public String actorName(long id) {
        return Optional.ofNullable(users.displayName(id)).orElse("已删除账号");
    }

    private void orphan(String id) {
        blobs.orphan(id);
    }

    private long team(CanvasActor actor) {
        return actor.teamId() == null ? 0 : actor.teamId();
    }

    private Job map(CanvasExportEntity row) {
        return new Job(row.getId(), row.getCanvasId(), row.getNodeRunId(),
                Format.valueOf(row.getFormat()), row.getStatus(),
                json.read(row.getContextJson(), ResourceCommandContext.class),
                json.read(row.getManifestJson(), Manifest.class),
                row.getManifestHash(), row.getCancelRequested(), row.getRevision(),
                row.getAttempts(),
                row.getWorkerId(), row.getOwnerEpoch(), instant(row.getLeaseUntil()),
                row.getBlobId(),
                row.getFilename(), row.getMimeType(), row.getByteSize(),
                row.getContentHash(),
                row.getLastError(), instant(row.getCreateTime()),
                instant(row.getUpdateTime()));
    }

    private Instant instant(LocalDateTime value) {
        return value == null ? null : value.toInstant(ZoneOffset.UTC);
    }

    static MediaBlob blob(CanvasExportBlobEntity row) {
        return new MediaBlob(row.getId(), row.getExportId(), row.getStorageConfigId(), row.getStorageBackend(),
                row.getObjectKey(), row.getEncryptionKeyset(), row.getStoredBytes());
    }
}
