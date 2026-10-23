package com.stonewu.fusion.service.media;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.media.model.*;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;

/**
 * 同一个来源对象/版本/字段对应一个托管资源；继承来源归属，不能从 URL 或当前团队猜测。
 */
@Service
@RequiredArgsConstructor
public class MediaSourceImportService {
    private final MediaSourceQueries sources;
    private final MediaSourceFetcher fetcher;
    private final MediaProbe probe;
    private final MediaResourceRepository resources;
    private final MediaRegistryTransactions registry;
    private final MediaBlobStore blobs;

    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaResource importSource(MediaSourceReference reference,
                                      ResourceCommandContext context) throws IOException {
        MediaIoBoundary.requireOutsideTransaction();
        var snapshot = sources.resolve(reference, context);
        return receive(snapshot, context, null, null);
    }

    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaResource importJobSource(MediaSourceReference reference, ResourceCommandContext context,
                                         MediaJobOrigin origin, Runnable check) throws IOException {
        MediaIoBoundary.requireOutsideTransaction();
        check.run();
        return receive(sources.resolve(reference, context), context, origin, check);
    }

    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaResource importJobSource(MediaSourceSnapshot snapshot, ResourceCommandContext context,
                                         MediaJobOrigin origin, Runnable check) throws IOException {
        MediaIoBoundary.requireOutsideTransaction();
        check.run();
        if (!snapshot.equals(sources.resolve(snapshot.reference(), context))) {
            throw new MediaException(409, "MEDIA_SOURCE_CHANGED", "导入来源已经变化，请按当前版本重新创建任务");
        }
        return receive(snapshot, context, origin, check);
    }

    private MediaResource receive(MediaSourceSnapshot snapshot, ResourceCommandContext context,
                                  MediaJobOrigin origin, Runnable check) throws IOException {
        if (MediaReference.managed(snapshot.uri())) {
            return origin == null ? registry.aliasSource(snapshot, context)
                    : registry.aliasJobSource(snapshot, context, origin);
        }
        var existing = registry.existingSource(snapshot, context);
        if (existing.isPresent()) {
            return existing.get();
        }
        var staging = Files.createTempFile("afv-source-import-", ".input");
        MediaIngestClaim claim = null;
        try {
            if (origin == null) {
                fetcher.copy(snapshot.uri(), staging);
            } else {
                fetcher.copy(snapshot.uri(), staging, check);
            }
            var facts = probe.inspect(staging, "application/octet-stream");
            claim = origin == null ? registry.admitSource(snapshot, facts, context) : registry.admitJobSource(snapshot,
                    facts, context, origin);
            if (claim.replay()) {
                return claim.resource();
            }
            long stored = blobs.write(resources.blob(claim.blobId()), staging);
            return registry.finish(claim, stored, context);
        } catch (IOException | RuntimeException error) {
            if (claim != null && !claim.replay()) {
                try {
                    registry.fail(claim);
                } catch (RuntimeException cleanup) {
                    error.addSuppressed(cleanup);
                }
            }
            throw error;
        } finally {
            Files.deleteIfExists(staging);
        }
    }
}
