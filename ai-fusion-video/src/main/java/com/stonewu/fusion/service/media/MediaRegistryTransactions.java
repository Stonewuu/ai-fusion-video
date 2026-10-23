package com.stonewu.fusion.service.media;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.stonewu.fusion.entity.media.MediaBlobEntity;
import com.stonewu.fusion.entity.media.MediaResourceEntity;
import com.stonewu.fusion.entity.media.MediaPreviewJobEntity;
import com.stonewu.fusion.mapper.media.MediaResourceMapper;
import com.stonewu.fusion.mapper.media.MediaBlobMapper;
import com.stonewu.fusion.mapper.media.MediaPreviewJobMapper;
import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.media.MediaAccessTicketMapper;
import com.stonewu.fusion.mapper.storage.StorageConfigMapper;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.media.job.MediaJobFence;
import com.stonewu.fusion.service.media.model.*;
import com.stonewu.fusion.service.media.preview.MediaPreviewData;
import com.stonewu.fusion.service.media.preview.MediaPreviewOwnership;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.storage.StorageTypes;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/**
 * 接收对象前后以短事务验证身份、代数及租约；文件传输不持有数据库事务或连接。
 */
@Service
@RequiredArgsConstructor
public class MediaRegistryTransactions {
    private final MediaResourceMapper media;
    private final MediaBlobMapper blobs;
    private final MediaPreviewJobMapper previewJobs;
    private final MediaJobMapper mediaJobs;
    private final MediaAccessTicketMapper tickets;
    private final StorageConfigMapper configurations;
    private final UserMapper users;
    private final ExecutionCoordinationMapper clock;
    private final MediaResourceRepository resources;
    private final MediaAccessPolicy access;
    private final ResourceAgentFence agentFence;
    private final MediaCryptor cryptor;
    private final CanvasJson json;
    private final MediaSourceQueries sources;
    private final GenerationExecutionContexts generations;
    private final MediaJobFence jobs;
    private final MediaPreviewOwnership previews;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaResource aliasSource(MediaSourceSnapshot observed, ResourceCommandContext context) {
        agentFence.requireActive(context, observed.projectId());
        lockActor(context);
        var current = sources.lock(observed.reference(), context);
        if (!current.equals(observed)) {
            throw new MediaException(409, "MEDIA_SOURCE_CHANGED", "媒体来源已变化");
        }
        var resource = resources.get(MediaReference.id(current.uri()), true).resource();
        access.ready(resource, context.actor());
        if (resource.content().kind() != current.kind()) {
            throw MediaException.invalid("来源字段的媒体类别不符");
        }
        var previous = resources.source(current.reference(), true);
        if (previous.isPresent() && !previous.get().resource().id().equals(resource.id())) {
            throw new MediaException(409, "MEDIA_SOURCE_CHANGED", "同一来源版本已映射到不同的媒体");
        }
        if (previous.isEmpty()) {
            resources.bindSource(resource.id(), current.reference());
        }
        return resource;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaResource aliasJobSource(MediaSourceSnapshot observed, ResourceCommandContext context,
                                        MediaJobOrigin origin) {
        jobs.require(origin.claim(), context, false);
        return aliasSource(observed, context);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Optional<MediaResource> existingSource(MediaSourceSnapshot observed, ResourceCommandContext context) {
        agentFence.requireActive(context, observed.projectId());
        lockActor(context);
        var source = sources.lock(observed.reference(), context);
        if (!source.equals(observed)) {
            throw new MediaException(409, "MEDIA_SOURCE_CHANGED", "媒体来源已变化，请刷新后重新导入");
        }
        var existing = resources.source(source.reference(), true);
        if (existing.isEmpty()) {
            return Optional.empty();
        }
        var row = existing.get();
        access.read(row.resource(), context.actor());
        String expected = sourceHash(source, row.resource().content());
        if (!expected.equals(row.requestHash())) {
            throw new MediaException(409, "MEDIA_SOURCE_CHANGED", "同一业务版本内的来源字段已变化，需要更新来源版本");
        }
        if ("READY".equals(row.resource().state())) {
            return Optional.of(row.resource());
        }
        LocalDateTime now = clock.now();
        if ("STAGING".equals(row.resource().state()) && row.leaseUntil() != null && row.leaseUntil().isAfter(now)) {
            throw new MediaException(409, "MEDIA_INGEST_IN_PROGRESS", "此来源仍在接收中，请稍后重试");
        }
        return Optional.empty();
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaIngestClaim admit(String requestId, Long projectId, String name, MediaFacts facts,
                                  ResourceCommandContext context) {
        validateRequest(requestId);
        if (requestId.startsWith("source:") || requestId.startsWith("generation:") || requestId.startsWith("job:")) {
            throw MediaException.invalid("此幂等键前缀由媒体来源登记保留");
        }
        agentFence.requireActive(context, projectId);
        lockActor(context);
        var owner = access.creation(projectId, context);
        String digest = json.hash(new IntakeIdentity(name, facts, projectId, context));
        return receive(requestId, projectId, owner, name, facts, digest,
                resources.request(context.actor().userId(), requestId), context, null, null);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaIngestClaim admitGeneration(MediaGenerationOrigin origin, String slot, String name, MediaFacts facts,
                                            ResourceCommandContext context) {
        validateRequest(slot);
        var observed = generations.get(origin.kind(), origin.taskId());
        agentFence.requireActive(context, observed.projectId());
        lockActor(context);
        var current = generations.lockActive(origin.kind(), origin.taskId(), context);
        var owner = access.creation(current.projectId(), context);
        String requestId = "generation:" + origin.kind().name().toLowerCase() + ":" + origin.taskId() + ":" + slot;
        validateRequest(requestId);
        String digest = json.hash(new IntakeIdentity(name, facts, current.projectId(), context));
        var claim = receive(requestId, current.projectId(), owner, name, facts, digest,
                resources.request(context.actor().userId(), requestId), context, null, null);
        return new MediaIngestClaim(claim.resource(), claim.blobId(), claim.epoch(), claim.replay(),
                claim.actorUserId(), null, origin);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaIngestClaim admitJob(MediaJobOrigin origin, String slot, Long projectId, String name, MediaFacts facts,
                                     ResourceCommandContext context) {
        jobs.require(origin.claim(), context, false);
        Long actualProject = mediaJobs.selectById(origin.claim().id()).getProjectId();
        if (!Objects.equals(projectId, actualProject)) {
            throw MediaException.invalid("任务产物的项目归属不符");
        }
        String requestId = "job:" + origin.claim().id() + ":" + slot;
        validateRequest(requestId);
        var owner = access.creation(projectId, context);
        String digest = json.hash(new IntakeIdentity(name, facts, projectId, context));
        var claim = receive(requestId, projectId, owner, name, facts, digest,
                resources.request(context.actor().userId(), requestId), context, null, null);
        return new MediaIngestClaim(claim.resource(), claim.blobId(), claim.epoch(), claim.replay(),
                claim.actorUserId(), null, null, origin);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaIngestClaim admitSource(MediaSourceSnapshot observed, MediaFacts facts,
                                        ResourceCommandContext context) {
        agentFence.requireActive(context, observed.projectId());
        lockActor(context);
        var source = sources.lock(observed.reference(), context);
        if (!source.equals(observed)) {
            throw new MediaException(409, "MEDIA_SOURCE_CHANGED", "媒体来源已变化，请刷新后重新导入");
        }
        if (facts.kind() != source.kind()) {
            throw MediaException.invalid("来源字段的媒体类别与实际内容不符");
        }
        String requestId = "source:" + json.hash(source.reference());
        String digest = sourceHash(source, facts);
        return receive(requestId, source.projectId(), new MediaAccessPolicy.Owner(source.ownerType(), source.ownerId()),
                source.name() + extension(facts.mimeType()), facts, digest, resources.source(source.reference(), true),
                context, source, null);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaIngestClaim admitJobSource(MediaSourceSnapshot observed, MediaFacts facts,
                                           ResourceCommandContext context, MediaJobOrigin origin) {
        jobs.require(origin.claim(), context, false);
        var claim = admitSource(observed, facts, context);
        return new MediaIngestClaim(claim.resource(), claim.blobId(), claim.epoch(), claim.replay(),
                claim.actorUserId(), claim.source(), null, origin);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaIngestClaim admitPreview(MediaJobOrigin origin, MediaFacts facts, ResourceCommandContext context) {
        jobs.require(origin.claim(), context, false);
        var source = previews.source(origin.claim().id(), context, true);
        if (facts.kind() != MediaKind.IMAGE || !facts.mimeType()
                .equals("image/png") || facts.width() == null || facts.height() == null
                || facts.width() <= 0 || facts.height() <= 0 || facts.width() > MediaPreviewData.WIDTH || facts.height() > MediaPreviewData.HEIGHT
                || facts.byteSize() <= 0 || facts.byteSize() > 3 * 1024 * 1024) {
            throw MediaException.invalid("生成的预览不符合已登记规格");
        }
        String requestId = "job:" + origin.claim().id() + ":preview";
        var owner = new MediaAccessPolicy.Owner(source.ownerType(), source.ownerId());
        String hash = json.hash(
                Map.of("source", source.id(), "version", source.version(), "profile", MediaPreviewData.PROFILE,
                        "content", facts));
        var existing = resources.request(context.actor().userId(), requestId);
        if (existing.isPresent() && existing.get().resource().state().equals("STAGING")) {
            Long receiverEpoch = mediaJobs.previewIngestEpoch(origin.claim().id());
            if (receiverEpoch == null) {
                throw MediaException.invalid("预览接收记录缺少执行代数");
            }
            if (receiverEpoch > origin.claim().epoch()) {
                throw MediaJobFence.lost();
            }
            if (receiverEpoch < origin.claim().epoch()) {
                // 原任务代数已失效，旧接收者不可能发布；无需等通用上传的十分钟租约。
                media.expireIngest(existing.get().resource().id());
                existing = resources.request(context.actor().userId(), requestId);
            }
        }
        var claim = receive(requestId, source.projectId(), owner, "预览.png", facts, hash,
                existing, context, null, source);
        int changed = previewJobs.update(Wrappers.<MediaPreviewJobEntity>lambdaUpdate()
                .eq(MediaPreviewJobEntity::getJobId, origin.claim().id())
                .and(condition -> condition.isNull(MediaPreviewJobEntity::getOutputResourceId)
                        .or().eq(MediaPreviewJobEntity::getOutputResourceId, claim.resource().id()))
                .set(MediaPreviewJobEntity::getOutputResourceId, claim.resource().id()));
        if (changed != 1) {
            throw MediaException.invalid("预览任务已有不同的产物");
        }
        mediaJobs.previewIngestOwner(origin.claim().epoch(), origin.claim().id());
        return new MediaIngestClaim(claim.resource(), claim.blobId(), claim.epoch(), claim.replay(),
                claim.actorUserId(), null, null, origin);
    }

    private MediaIngestClaim receive(String requestId, Long projectId, MediaAccessPolicy.Owner owner, String name,
                                     MediaFacts facts,
                                     String digest, Optional<MediaResourceRepository.Row> existing,
                                     ResourceCommandContext context, MediaSourceSnapshot source,
                                     MediaResource previewSource) {
        MediaResource resource;
        String blobId = UUID.randomUUID().toString();
        long epoch = 1;
        if (existing.isPresent()) {
            var previous = existing.get();
            resource = previous.resource();
            access.read(resource, context.actor());
            if (source == null && previewSource == null) {
                access.edit(resource, context);
            } else if (!Objects.equals(resource.projectId(),
                    projectId) || resource.ownerType() != owner.type() || resource.ownerId() != owner.id()) {
                throw new MediaException(409, "MEDIA_SOURCE_CHANGED", "来源归属与已登记媒体不一致");
            }
            if (!previous.requestHash().equals(digest)) {
                throw new MediaException(409, "MEDIA_REQUEST_CONFLICT", "幂等键已用于不同的媒体内容或作用域");
            }
            if ("READY".equals(resource.state())) {
                return new MediaIngestClaim(resource, previous.blobId(), previous.epoch(), true,
                        context.actor().userId(), source);
            }
            LocalDateTime now = clock.now();
            if ("STAGING".equals(resource.state()) && previous.leaseUntil() != null && previous.leaseUntil()
                    .isAfter(now)) {
                throw new MediaException(409, "MEDIA_INGEST_IN_PROGRESS", "此文件仍在接收中，请稍后使用同一幂等键重试");
            }
            epoch = previous.epoch() + 1;
            blobs.orphanStaging(previous.blobId());
            media.restartIngest(blobId, epoch, resource.id());
        } else {
            String id = UUID.randomUUID().toString();
            media.insertStaging(MediaResourceEntity.builder()
                    .id(id)
                    .projectId(projectId)
                    .ownerType(owner.type())
                    .ownerId(owner.id())
                    .createdBy(context.actor().userId())
                    .actorTeamId(context.actor().teamId())
                    .requestId(requestId)
                    .requestHash(digest)
                    .mediaKind(facts.kind().name())
                    .mimeType(facts.mimeType())
                    .originalName(name)
                    .byteSize(facts.byteSize())
                    .contentHash(facts.contentHash())
                    .width(facts.width())
                    .height(facts.height())
                    .durationMs(facts.durationMs())
                    .hasAudio(facts.hasAudio())
                    .blobId(blobId)
                    .origin(context.origin())
                    .agentRunId(context.agentRunId())
                    .toolCallId(context.toolCallId())
                    .canvasId(context.canvasId()).build());
            resource = resources.get(id, false).resource();
            if (source != null) {
                resources.bindSource(id, source.reference());
            }
        }
        var storage = configurations.lockActiveDefaults();
        if (storage.size() != 1) {
            throw MediaException.unavailable("请设置且只设置一个启用的默认存储配置");
        }
        StorageConfig config = storage.getFirst();
        String backend = StorageTypes.normalizeType(config.getType());
        if (!"local".equals(backend) && !"s3".equals(backend)) {
            throw MediaException.unavailable("默认存储后端不支持托管媒体");
        }
        if ("local".equals(backend) && (config.getBasePath() == null || config.getBasePath().toString()
                .isBlank())) {
            throw MediaException.unavailable("本地媒体存储必须明确配置根目录");
        }
        blobs.insert(MediaBlobEntity.builder().id(blobId).resourceId(resource.id()).ingestEpoch(epoch)
                .storageConfigId(config.getId()).backend(backend).objectKey(blobId + ".afvblob")
                .keysetBytes(cryptor.newKeyset()).state("STAGING").build());
        resource = resources.get(resource.id(), false).resource();
        resources.event(resource, "media.staging", context.actor().userId());
        agentFence.requireActive(context, projectId);
        return new MediaIngestClaim(resource, blobId, epoch, false, context.actor().userId(), source);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaResource finish(MediaIngestClaim claim, long storedBytes, ResourceCommandContext context) {
        if (storedBytes <= 0) {
            throw MediaException.invalid("媒体存储结果大小无效");
        }
        agentFence.requireActive(context, claim.resource().projectId());
        lockActor(context);
        if (claim.actorUserId() != context.actor().userId()) {
            throw MediaException.invalid("接收结果不属于当前操作用户");
        }
        if (claim.job() != null) {
            jobs.require(claim.job().claim(), context, false);
        }
        if (claim.generation() != null) {
            generations.lockActive(claim.generation().kind(), claim.generation().taskId(), context);
        }
        if (claim.source() != null && !sources.lock(claim.source().reference(), context).equals(claim.source())) {
            throw new MediaException(409, "MEDIA_SOURCE_CHANGED", "媒体接收期间来源已变化，请刷新后重新导入");
        }
        var row = resources.get(claim.resource().id(), true);
        boolean previewOutput = claim.job() != null && previewJobs.selectList(
                Wrappers.<MediaPreviewJobEntity>lambdaQuery()
                        .select(MediaPreviewJobEntity::getJobId)
                        .eq(MediaPreviewJobEntity::getJobId, claim.job().claim().id())
                        .eq(MediaPreviewJobEntity::getOutputResourceId, row.resource().id())).size() == 1;
        if (resources.previewParent(row.resource().id()).isPresent() && !previewOutput) {
            throw MediaException.invalid("预览产物必须由原预览任务提交");
        }
        if (claim.source() == null && !previewOutput) {
            access.edit(row.resource(), context);
        } else {
            access.read(row.resource(), context.actor());
        }
        if (!Objects.equals(row.blobId(), claim.blobId()) || row.epoch() != claim.epoch() || !"STAGING".equals(
                row.resource().state())) {
            throw new MediaException(409, "MEDIA_INGEST_FENCED", "此上传已由新的接收任务接管");
        }
        int changed = media.finishIngest(row.resource().id());
        if (changed != 1) {
            throw new MediaException(409, "MEDIA_INGEST_FENCED", "上传租约已过期，请使用同一幂等键重试");
        }
        blobs.update(Wrappers.<MediaBlobEntity>lambdaUpdate().eq(MediaBlobEntity::getId, claim.blobId())
                .eq(MediaBlobEntity::getState, "STAGING").set(MediaBlobEntity::getState, "READY")
                .set(MediaBlobEntity::getStoredBytes, storedBytes));
        var resource = resources.get(row.resource().id(), false).resource();
        resources.event(resource, "media.ready", context.actor().userId());
        agentFence.requireActive(context, resource.projectId());
        return resource;
    }

    /**
     * 清理失效接收者的状态；只允许其自身代数结束，不修改任何业务对象或新接收者的结果。
     */
    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaResource", allEntries = true)
    public void fail(MediaIngestClaim claim) {
        var row = resources.get(claim.resource().id(), true);
        if (!"STAGING".equals(row.resource().state()) || row.epoch() != claim.epoch()
                || !Objects.equals(row.blobId(), claim.blobId())) {
            return;
        }
        media.failIngest(row.resource().id());
        blobs.orphan(claim.blobId());
        resources.event(resources.get(row.resource().id(), false).resource(), "media.failed", claim.actorUserId());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaResource", allEntries = true)
    public int expireUploads() {
        var expired = media.lockExpiredIngests();
        for (String id : expired) {
            var row = resources.get(id, false);
            fail(new MediaIngestClaim(row.resource(), row.blobId(), row.epoch(), false));
        }
        tickets.removeExpired();
        return expired.size();
    }

    private void lockActor(ResourceCommandContext context) {
        var user = users.lockEnabled(context.actor().userId());
        if (user == null) {
            throw new MediaException(403, "MEDIA_ACCESS_DENIED", "当前用户不可用");
        }
    }

    private void validateRequest(String id) {
        if (id == null || !id.matches("[A-Za-z0-9._:-]{1,128}")) {
            throw MediaException.invalid("媒体幂等键格式无效");
        }
    }

    private String sourceHash(MediaSourceSnapshot source, MediaFacts facts) {
        return MediaSourceIdentity.hash(json, source, facts);
    }

    private String extension(String mime) {
        return switch (mime) {
            case "image/png" -> ".png";
            case "image/jpeg" -> ".jpg";
            case "image/gif" -> ".gif";
            case "image/webp" -> ".webp";
            case "video/mp4" -> ".mp4";
            case "audio/mp4" -> ".m4a";
            case "video/quicktime" -> ".mov";
            case "video/webm", "audio/webm" -> ".webm";
            case "video/x-matroska", "audio/x-matroska" -> ".mkv";
            case "video/3gpp", "audio/3gpp" -> ".3gp";
            case "video/3gpp2", "audio/3gpp2" -> ".3g2";
            case "audio/mpeg" -> ".mp3";
            case "audio/wav" -> ".wav";
            case "audio/flac" -> ".flac";
            case "audio/ogg", "video/ogg" -> ".ogg";
            case "audio/aac" -> ".aac";
            case "video/mpeg" -> ".mpeg";
            default -> throw MediaException.invalid("媒体来源类型没有文件名映射");
        };
    }

    // 隔离代数、租约持有者不进入内容身份；合法的 Agent 接管可重放相同工具操作。
    private record IntakeIdentity(String name, MediaFacts facts, Long projectId, String origin, long actorUserId,
                                  Long actorTeamId, String canvasId, String agentRunId, String toolCallId) {
        IntakeIdentity(String name, MediaFacts facts, Long projectId, ResourceCommandContext context) {
            this(name, facts, projectId, context.origin(), context.actor().userId(), context.actor().teamId(),
                    context.canvasId(), context.agentRunId(), context.toolCallId());
        }
    }
}
