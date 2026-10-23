package com.stonewu.fusion.service.media.preview;

import com.stonewu.fusion.mapper.media.MediaPreviewJobMapper;
import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobFence;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;
import java.util.Optional;

import static com.stonewu.fusion.service.media.preview.MediaPreviewData.*;

@Service
@RequiredArgsConstructor
public class MediaPreviewJobs {
    private final MediaPreviewJobMapper previews;
    private final MediaJobMapper mediaJobs;
    private final ExecutionCoordinationMapper coordination;
    private final CanvasJson json;
    private final MediaJobRepository jobs;
    private final MediaJobFence fence;
    private final MediaPreviewService service;
    private final MediaResourceRepository resources;
    private final MediaAccessPolicy access;
    private final MediaPreviewOwnership ownership;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Optional<MediaJob> claim(String owner) {
        coordination.lockGenerationDispatch();
        if (mediaJobs.activePreviews() >= 4) {
            return Optional.empty();
        }
        var ids = mediaJobs.nextPreview();
        if (ids.isEmpty()) {
            return Optional.empty();
        }
        mediaJobs.claimPreview(owner, ids.getFirst());
        var job = jobs.get(ids.getFirst());
        service.event(job, "MEDIA_PREVIEW_CHANGED");
        return Optional.of(job);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public MediaJob checked(MediaJob.Claim claim) {
        var job = jobs.get(claim.id());
        if (!KIND.equals(job.kind())) {
            throw MediaJobFence.lost();
        }
        fence.require(claim, job.context(), false);
        return jobs.get(job.id());
    }

    public Input input(MediaJob job) {
        var input = json.read(json.write(job.input()), Input.class);
        if (input.schemaVersion() != 1 || !PROFILE.equals(
                input.profile()) || input.source() == null || input.sampleAtMs() < 0) {
            throw MediaException.invalid("预览任务的输入协议无效");
        }
        if (!input.source().equals(ownership.source(job.id(), job.context(),
                false)) || input.sampleAtMs() != MediaPreviewQueries.sample(input.source())) {
            throw MediaException.invalid("预览任务输入与固定的原素材不一致");
        }
        return input;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Optional<String> existing(MediaJob.Claim claim) {
        var job = checked(claim);
        String id = previews.selectById(claim.id()).getOutputResourceId();
        if (id == null) {
            return Optional.empty();
        }
        var resource = resources.get(id, false).resource();
        if (!resource.state().equals("READY")) {
            return Optional.empty();
        }
        access.ready(resource, job.context().actor());
        return Optional.of(id);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public void complete(MediaJob.Claim claim, String resourceId) {
        var job = checked(claim);
        String output = previews.selectById(claim.id()).getOutputResourceId();
        if (!resourceId.equals(output)) {
            throw MediaException.invalid("预览产物不属于当前任务");
        }
        access.ready(resources.get(output, false).resource(), job.context().actor());
        mediaJobs.completePreview(json.write(Map.of("resourceId", resourceId)), job.id());
        service.event(jobs.get(job.id()), "MEDIA_PREVIEW_CHANGED");
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public void waitForIngest(MediaJob.Claim claim) {
        var job = checked(claim);
        mediaJobs.waitForPreviewIngest(job.id());
        service.event(jobs.get(job.id()), "MEDIA_PREVIEW_CHANGED");
    }

    /**
     * 失败收尾不发布产物，原权限消失后也只能结束自己仍持有的租约。
     */
    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public void stop(MediaJob.Claim claim, boolean cancelled, String message) {
        var found = mediaJobs.lockPreviewOwner(claim.id(), claim.owner(), claim.epoch());
        if (found.isEmpty()) {
            return;
        }
        mediaJobs.stopPreview(cancelled ? "CANCELLED" : "FAILED", cancelled ? "预览生成已停止" : "预览生成失败",
                message, claim.id());
        service.event(jobs.get(claim.id()), "MEDIA_PREVIEW_CHANGED");
    }
}
