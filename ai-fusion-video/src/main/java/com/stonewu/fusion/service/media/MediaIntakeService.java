package com.stonewu.fusion.service.media;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.media.model.*;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;

/**
 * 用户上传及内部产物接收的共同入口；只有完成探测、存储和提交之后才提供可用资源。
 */
@Service
@RequiredArgsConstructor
public class MediaIntakeService {
    public static final long MAX_UPLOAD_BYTES = 100L * 1024 * 1024;
    public static final long MAX_RESULT_BYTES = 2L * 1024 * 1024 * 1024;
    private final MediaAccessPolicy access;
    private final MediaProbe probe;
    private final MediaRegistryTransactions registry;
    private final MediaResourceRepository resources;
    private final MediaBlobStore blobs;

    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaResource upload(InputStream source, String name, String declaredMime, String requestId,
                                Long projectId, ResourceCommandContext context) throws IOException {
        MediaIoBoundary.requireOutsideTransaction();
        return receive(source, name, declaredMime, requestId, projectId, context, MAX_UPLOAD_BYTES, null, null, null);
    }

    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaResource result(Path source, String name, String declaredMime, String requestId,
                                Long projectId, ResourceCommandContext context) throws IOException {
        MediaIoBoundary.requireOutsideTransaction();
        try (var stream = Files.newInputStream(source)) {
            return receive(stream, name, declaredMime, requestId, projectId, context, MAX_RESULT_BYTES, null, null,
                    null);
        }
    }

    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaResource generation(Path source, String name, String declaredMime, String slot, Long projectId,
                                    ResourceCommandContext context, MediaGenerationOrigin origin) throws IOException {
        MediaIoBoundary.requireOutsideTransaction();
        try (var stream = Files.newInputStream(source)) {
            return receive(stream, name, declaredMime, slot, projectId, context, MAX_RESULT_BYTES, origin, null, null);
        }
    }

    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaResource job(Path source, String name, String declaredMime, String slot, Long projectId,
                             ResourceCommandContext context, MediaJobOrigin origin) throws IOException {
        MediaIoBoundary.requireOutsideTransaction();
        try (var stream = Files.newInputStream(source)) {
            return receive(stream, name, declaredMime, slot, projectId, context, MAX_RESULT_BYTES, null, origin, null);
        }
    }

    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaResource uploadChecked(InputStream source, String name, String declaredMime, String requestId,
                                       Long projectId, ResourceCommandContext context,
                                       Set<MediaKind> allowed) throws IOException {
        MediaIoBoundary.requireOutsideTransaction();
        if (allowed == null || allowed.isEmpty()) {
            throw MediaException.invalid("必须指定上传允许的媒体类型");
        }
        return receive(source, name, declaredMime, requestId, projectId, context, MAX_UPLOAD_BYTES, null, null,
                Set.copyOf(allowed));
    }

    private MediaResource receive(InputStream source, String name, String declaredMime, String requestId,
                                  Long projectId, ResourceCommandContext context, long limit,
                                  MediaGenerationOrigin generation, MediaJobOrigin job,
                                  Set<MediaKind> allowed) throws IOException {
        access.creation(projectId, context);
        String safeName = safeName(name);
        Path staging = Files.createTempFile("afv-media-intake-", ".input");
        MediaIngestClaim claim = null;
        try {
            long total = 0;
            try (var output = Files.newOutputStream(staging)) {
                byte[] buffer = new byte[65536];
                for (int count; (count = source.read(buffer)) != -1; ) {
                    total += count;
                    if (total > limit) {
                        throw new MediaException(413, "MEDIA_TOO_LARGE", "媒体文件超过大小限制");
                    }
                    output.write(buffer, 0, count);
                }
            }
            var facts = probe.inspect(staging, declaredMime);
            if (allowed != null && !allowed.contains(facts.kind())) {
                throw MediaException.invalid("实际媒体类型不适合此上传入口");
            }
            claim = job != null ? registry.admitJob(job, requestId, projectId, safeName, facts, context)
                    : generation != null ? registry.admitGeneration(generation, requestId, safeName, facts, context)
                    : registry.admit(requestId, projectId, safeName, facts, context);
            if (claim.replay()) {
                return claim.resource();
            }
            long bytes = blobs.write(resources.blob(claim.blobId()), staging);
            return registry.finish(claim, bytes, context);
        } catch (IOException | RuntimeException failure) {
            if (claim != null && !claim.replay()) {
                try {
                    registry.fail(claim);
                } catch (RuntimeException cleanup) {
                    failure.addSuppressed(cleanup);
                }
            }
            throw failure;
        } finally {
            Files.deleteIfExists(staging);
        }
    }

    private String safeName(String name) {
        if (name == null || name.isBlank()) {
            throw MediaException.invalid("文件名不能为空");
        }
        String safe = name.replaceAll("[\\p{Cntrl}/\\\\]", "_").trim();
        if (safe.length() > 255) {
            throw MediaException.invalid("文件名不能超过 255 个字符");
        }
        return safe;
    }
}
