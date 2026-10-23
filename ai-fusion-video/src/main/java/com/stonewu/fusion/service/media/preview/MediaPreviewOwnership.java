package com.stonewu.fusion.service.media.preview;

import com.stonewu.fusion.mapper.media.MediaPreviewJobMapper;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.job.MediaJobFence;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;

import java.util.Objects;

@Component
@RequiredArgsConstructor
public class MediaPreviewOwnership {
    private final MediaPreviewJobMapper previews;
    private final MediaResourceRepository resources;
    private final MediaAccessPolicy access;

    public MediaResource source(String jobId, ResourceCommandContext context, boolean lock) {
        var input = previews.source(jobId, MediaPreviewData.PROFILE);
        if (input == null) {
            throw MediaJobFence.lost();
        }
        var source = resources.get(input.id(), lock).resource();
        access.ready(source, context.actor());
        if (source.version() != input.version() || !Objects.equals(source.projectId(), input.projectId())
                || resources.previewParent(source.id()).isPresent()) {
            throw MediaJobFence.lost();
        }
        return source;
    }

}
