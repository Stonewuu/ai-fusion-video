package com.stonewu.fusion.service.media.preview;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.media.MediaBlobStore;
import com.stonewu.fusion.service.media.MediaIoBoundary;
import com.stonewu.fusion.service.media.MediaProbe;
import com.stonewu.fusion.service.media.MediaRegistryTransactions;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.model.MediaJobOrigin;
import com.stonewu.fusion.service.media.model.MediaResource;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Path;

@Service
@RequiredArgsConstructor
public class MediaPreviewIntake {
    private final MediaProbe probe;
    private final MediaRegistryTransactions registry;
    private final MediaResourceRepository resources;
    private final MediaBlobStore blobs;

    @CacheEvict(value = "mediaResource", allEntries = true)
    public MediaResource store(MediaJob job, Path file) throws IOException {
        MediaIoBoundary.requireOutsideTransaction();
        var facts = probe.inspect(file, "image/png");
        var claim = registry.admitPreview(new MediaJobOrigin(job.claim()), facts, job.context());
        if (claim.replay()) {
            return claim.resource();
        }
        try {
            long bytes = blobs.write(resources.blob(claim.blobId()), file);
            return registry.finish(claim, bytes, job.context());
        } catch (IOException | RuntimeException failure) {
            try {
                registry.fail(claim);
            } catch (RuntimeException cleanup) {
                failure.addSuppressed(cleanup);
            }
            throw failure;
        }
    }
}
