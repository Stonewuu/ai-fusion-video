package com.stonewu.fusion.service.canvas.run.media;

import com.stonewu.fusion.mapper.canvas.CanvasMediaJobMapper;
import com.stonewu.fusion.entity.canvas.CanvasMediaJobEntity;
import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@Service
@RequiredArgsConstructor
public class CanvasMediaReviewService {
    public record View(String id, String status, String phase, String error, long revision,
                       String providerTaskId, boolean remoteCapacityHeld, boolean cancelRequested,
                       List<String> actions) {
    }

    private final CanvasRunRepository runs;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final MediaJobRepository jobs;
    private final CanvasMediaJobMapper canvasJobs;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View detail(String runId, String nodeId, CanvasActor actor) {
        var run = runs.run(runId, false);
        access.requireView(documents.require(run.canvasId(), false), actor);
        var job = task(runId, nodeId);
        var submission = jobs.submission(job.id(), "audio");
        boolean held = Boolean.TRUE.equals(
                canvasJobs.selectById(job.id()).getPermitHeld());
        return new View(job.id(), job.status(), job.phase(), job.error(), job.revision(),
                submission.map(MediaJobRepository.Submission::providerId).orElse(null), held, job.cancelRequested(),
                List.of());
    }

    private MediaJob task(String runId, String nodeId) {
        var node = runs.node(nodeId, false);
        if (!node.runId().equals(runId) || !"MEDIA".equals(node.externalKind()) || node.externalId() == null) {
            throw CanvasRunException.invalid("此节点没有对应媒体任务");
        }
        return jobs.get(node.externalId());
    }

}
