package com.stonewu.fusion.service.canvas.run.media;

import com.stonewu.fusion.mapper.canvas.CanvasMediaJobMapper;
import com.stonewu.fusion.entity.canvas.CanvasMediaJobEntity;
import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeRun;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Ports;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.generation.job.GenerationModelSnapshot;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobFence;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class CanvasMediaJobs {
    private final CanvasMediaJobMapper canvasJobs;
    private final MediaJobMapper mediaJobs;
    private final ExecutionCoordinationMapper coordination;
    private final CanvasJson json;
    private final MediaJobRepository jobs;
    private final MediaJobFence fence;
    private final GenerationOutbox outbox;
    private final GenerationModelSnapshot apis;
    private final MediaAccessPolicy access;
    private final CanvasRunSources sources;

    @Transactional(propagation = Propagation.MANDATORY)
    @CacheEvict(value = "canvasMediaJob", allEntries = true)
    public String submit(Run run, NodeRun node) {
        var input = new CanvasMediaInput(1, node.id(), node.plan().node().kind(), node.plan().node().config(),
                node.inputs(), node.plan().environment());
        input.inputs().all().forEach(value -> sources.authorize(value, run.actor(), true));
        var context = ResourceCommandContext.user(run.actor(), run.canvasId());
        access.creation(run.plan().graph().projectId(), context);
        String id = UUID.randomUUID().toString();
        mediaJobs.insertCanvasMedia(id, run.plan().graph().projectId(), run.actor().userId(),
                "canvas-media:" + node.id(), json.hash(input), json.write(context), json.write(input));
        var environment = node.plan().environment();
        canvasJobs.insert(
                CanvasMediaJobEntity.builder().jobId(id).nodeRunId(node.id()).operation(input.operation().code())
                        .apiConfigId(environment.has("apiConfigId") ? environment.path("apiConfigId").asLong() : null)
                        .apiConfigHash(environment.has("apiHash") ? environment.path("apiHash").asText() : null)
                        .build());
        event(jobs.get(id), "CANVAS_MEDIA_ACCEPTED", Map.of());
        return id;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Optional<MediaJob> claim(String owner) {
        coordination.lockGenerationDispatch();
        var ids = mediaJobs.nextCanvasMedia();
        if (ids.isEmpty()) {
            return Optional.empty();
        }
        String id = ids.getFirst();
        mediaJobs.claimCanvasMedia(owner, id);
        canvasJobs.acquirePermit(id);
        return Optional.of(jobs.get(id));
    }

    public boolean heartbeat(MediaJob.Claim claim) {
        return mediaJobs.heartbeat(claim.id(), claim.owner(), claim.epoch()) == 1;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public MediaJob check(MediaJob.Claim claim, boolean allowCancel) {
        var job = jobs.get(claim.id());
        fence.require(claim, job.context(), allowCancel);
        return jobs.get(claim.id());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasMediaJob", key = "#claim.id()")
    public void checkpoint(MediaJob.Claim claim, ObjectNode checkpoint, String phase, boolean allowCancel) {
        var job = check(claim, allowCancel);
        mediaJobs.checkpoint(json.write(checkpoint), phase, job.id());
        event(jobs.get(job.id()), "CANVAS_MEDIA_CHANGED", Map.of());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasMediaJob", key = "#claim.id()")
    public void waitNext(MediaJob.Claim claim, String phase, boolean failedQuery) {
        var job = check(claim, true);
        canvasJobs.pollResult(job.id(), failedQuery);
        int errors = canvasJobs.selectById(job.id()).getConsecutiveErrors();
        if (errors >= 8) {
            finish(claim, "NEEDS_REVIEW", "连续查询失败，请核实原音频任务", null, true);
            return;
        }
        mediaJobs.waitForCanvasMedia(phase, Math.min(30, 3 + errors * 3), job.id());
        event(jobs.get(job.id()), "CANVAS_MEDIA_CHANGED", Map.of());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasMediaJob", key = "#claim.id()")
    public MediaJobRepository.Submission beginAudio(MediaJob.Claim claim) {
        var job = check(claim, false);
        api(input(job));
        canvasJobs.update(Wrappers.<CanvasMediaJobEntity>lambdaUpdate().eq(CanvasMediaJobEntity::getJobId, job.id())
                .set(CanvasMediaJobEntity::getPermitHeld, true));
        return jobs.prepareSubmission(claim, "audio");
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasMediaJob", key = "#claim.id()")
    public void received(MediaJob.Claim claim, ObjectNode result) {
        check(claim, true);
        canvasJobs.update(Wrappers.<CanvasMediaJobEntity>lambdaUpdate().eq(CanvasMediaJobEntity::getJobId, claim.id())
                .set(CanvasMediaJobEntity::getResultJson, json.write(result))
                .set(CanvasMediaJobEntity::getPermitHeld, false)
                .set(CanvasMediaJobEntity::getConsecutiveErrors, 0));
    }

    public ObjectNode received(String id) {
        String result = canvasJobs.selectById(id).getResultJson();
        return result == null ? null : json.read(result, ObjectNode.class);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasMediaJob", key = "#claim.id()")
    public void finish(MediaJob.Claim claim, String status, String message, Ports output, boolean holdPermit) {
        var job = jobs.get(claim.id());
        if (output != null) {
            check(claim, false);
            input(job).inputs().all().forEach(value -> sources.authorize(value, job.context().actor(), true));
            output.all().forEach(value -> sources.authorize(value, job.context().actor(), true));
        }
        var checkpoint = job.checkpoint().deepCopy();
        if (output != null) {
            checkpoint.set("outputs", json.object(output));
        }
        int changed = mediaJobs.finishCanvasMedia(status, status.equals("SUCCEEDED") ? "处理完成" : message, message,
                json.write(checkpoint), claim.id(), claim.owner(), claim.epoch());
        if (changed != 1) {
            throw MediaJobFence.lost();
        }
        canvasJobs.update(Wrappers.<CanvasMediaJobEntity>lambdaUpdate().eq(CanvasMediaJobEntity::getJobId, job.id())
                .set(CanvasMediaJobEntity::getPermitHeld, holdPermit));
        event(jobs.get(job.id()), "CANVAS_MEDIA_CHANGED", Map.of());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasMediaJob", key = "#id")
    public void cancel(String id, CanvasActor actor) {
        var job = jobs.get(id);
        if (!job.context().actor().equals(actor)) {
            throw CanvasRunException.invalid("不能跨作用域取消媒体任务");
        }
        mediaJobs.cancelCanvasMedia(id);
    }

    public ApiConfig api(CanvasMediaInput input) {
        var api = apis.requireApi(input.environment().path("apiConfigId").asLong());
        if (!Objects.equals(apis.fingerprint(api), input.environment().path("apiHash").asText())) {
            throw CanvasRunException.conflict("音频供应商连接已变化，需要核实原任务");
        }
        return api;
    }

    public CanvasMediaInput input(MediaJob job) {
        return json.read(job.input().toString(), CanvasMediaInput.class);
    }

    private void event(MediaJob job, String type, Map<String, Object> extra) {
        var payload = json.object(extra).put("jobId", job.id()).put("revision", job.revision())
                .put("status", job.status());
        outbox.enqueue("canvas-media:" + job.id() + ":" + job.revision(), type, job.id(), payload);
    }
}
