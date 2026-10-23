package com.stonewu.fusion.service.media.preview;

import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.mapper.media.MediaPreviewJobMapper;
import com.stonewu.fusion.entity.media.MediaPreviewJobEntity;
import com.stonewu.fusion.mapper.media.MediaJobMapper;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import static com.stonewu.fusion.service.media.preview.MediaPreviewData.*;

@Service
@RequiredArgsConstructor
public class MediaPreviewService {
    private final UserMapper users;
    private final MediaPreviewJobMapper previews;
    private final MediaJobMapper mediaJobs;
    private final CanvasJson json;
    private final MediaPreviewQueries queries;
    private final MediaJobRepository jobs;
    private final GenerationOutbox outbox;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public View ensure(String sourceId, Request request, CanvasActor actor) {
        if (users.lockEnabled(actor.userId()) == null) {
            throw new MediaException(403, "MEDIA_ACCESS_DENIED", "当前账号不可用");
        }
        var source = queries.source(sourceId, actor, true);
        var current = previews.current(sourceId, source.version(), PROFILE, true);
        String previousId = null;
        if (current != null) {
            var previous = jobs.get(current);
            if (request.retryJobId() == null) {
                return queries.view(previous.id(), actor);
            }
            String predecessor = previews.selectById(previous.id()).getPreviousJobId();
            if (request.retryJobId().equals(predecessor)) {
                return queries.view(previous.id(), actor);
            }
            if (!request.retryJobId().equals(previous.id()) || !List.of("FAILED", "CANCELLED")
                    .contains(previous.status())) {
                throw changed();
            }
            previousId = previous.id();
        } else if (request.retryJobId() != null) {
            throw changed();
        }
        String id = UUID.randomUUID().toString();
        long sample = MediaPreviewQueries.sample(source);
        var input = new Input(1, source, PROFILE, sample);
        var context = new ResourceCommandContext(actor, "system", null, null, null, null, null);
        mediaJobs.insertPreview(id, source.projectId(), actor.userId(), "preview:" + id,
                json.hash(input), json.write(context), json.write(input));
        previews.insert(MediaPreviewJobEntity.builder().jobId(id).sourceResourceId(source.id())
                .sourceVersion(source.version()).profile(PROFILE).sampleAtMs(sample).previousJobId(previousId).build());
        previews.setCurrent(source.id(), source.version(), PROFILE, id);
        event(jobs.get(id), "MEDIA_PREVIEW_ACCEPTED");
        return queries.view(id, actor);
    }

    public void event(MediaJob job, String type) {
        outbox.enqueue("media-preview:" + job.id() + ":" + job.revision(), type, job.id(),
                Map.of("jobId", job.id(), "revision", job.revision(), "status", job.status()));
    }

    private MediaException changed() {
        return new MediaException(409, "MEDIA_PREVIEW_CHANGED", "预览状态已变化，请刷新后再重试");
    }
}
