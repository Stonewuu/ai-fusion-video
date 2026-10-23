package com.stonewu.fusion.service.canvas.run.text;

import com.stonewu.fusion.mapper.canvas.CanvasTextJobMapper;
import com.stonewu.fusion.entity.canvas.CanvasTextJobEntity;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;

import java.time.LocalDateTime;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeRun;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Repository;

import java.time.Instant;
import java.time.ZoneOffset;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static com.stonewu.fusion.service.canvas.run.text.CanvasTextData.*;

/**
 * 调用者负责事务与授权；任务结果与提交意图独立于工作者租约保存。
 */
@Repository
@RequiredArgsConstructor
public class CanvasTextRepository {
    private final CanvasTextJobMapper textJobs;
    private final ExecutionCoordinationMapper coordination;
    private final CanvasJson json;
    private final GenerationOutbox outbox;

    public Job get(String id, boolean lock) {
        var row = lock ? textJobs.lock(id) : textJobs.selectById(id);
        if (row == null) {
            throw new CanvasRunException(404, "TEXT_JOB_NOT_FOUND", "文本任务不存在");
        }
        return read(row);
    }

    public String insert(Run run, NodeRun node, Input input) {
        String id = UUID.randomUUID().toString();
        textJobs.insertJob(id, node.id(), run.id(), run.plan().graph().projectId(), input.model().getId(),
                input.model().getMaxConcurrency(), json.write(input),
                run.deadlineAt().atOffset(ZoneOffset.UTC).toLocalDateTime());
        event(get(id, false), "CANVAS_TEXT_ACCEPTED", Map.of());
        return id;
    }

    public Optional<Job> claim(String owner) {
        coordination.lockGenerationDispatch();
        var ids = textJobs.next();
        if (ids.isEmpty()) {
            return Optional.empty();
        }
        String id = ids.getFirst();
        textJobs.claim(owner, id);
        return Optional.of(get(id, false));
    }

    public Job leased(Claim claim) {
        var values = textJobs.leased(claim.id(), claim.owner(), claim.epoch());
        if (values == null) {
            throw CanvasRunException.fenced();
        }
        return read(values);
    }

    public boolean heartbeat(Claim claim) {
        return textJobs.heartbeat(claim.id(), claim.owner(), claim.epoch()) == 1;
    }

    public void begin(Claim claim) {
        if (textJobs.begin(claim.owner(), claim.epoch(), claim.id()) != 1) {
            throw CanvasRunException.conflict("原文本请求已经开始，不能重复提交");
        }
        event(get(claim.id(), false), "CANVAS_TEXT_CHANGED", Map.of());
    }

    public void received(Claim claim, ObjectNode result) {
        // 原工作者即使已失去租约，仍可登记原提交的迟到响应；不能发布结果或启动下游。
        var row = get(claim.id(), true);
        String hash = json.hash(result);
        if (row.result() != null) {
            if (!json.hash(row.result()).equals(hash)) {
                throw CanvasRunException.conflict("同一文本提交返回了不一致的响应");
            }
            return;
        }
        int changed = textJobs.received(json.write(result), hash, claim.id(), claim.owner(), claim.epoch());
        if (changed != 1) {
            throw CanvasRunException.fenced();
        }
        event(get(claim.id(), false), "CANVAS_TEXT_RECEIVED", Map.of());
    }

    public void finish(Job job, String state, String message, boolean permit) {
        String phase = switch (state) {
            case "SUCCEEDED" -> "文本生成完成";
            case "NEEDS_REVIEW", "UNKNOWN_SUBMISSION" -> "等待核实原调用";
            case "CANCELLED" -> "文本任务已停止";
            default -> "文本生成未完成";
        };
        textJobs.finish(state, phase, message, permit, job.id());
        event(get(job.id(), false), "CANVAS_TEXT_CHANGED", Map.of());
    }

    public void cancel(String id) {
        textJobs.cancel(id);
    }

    public void resume(Job job, Map<String, Object> audit) {
        textJobs.resume(job.id());
        event(get(job.id(), false), "CANVAS_TEXT_REVIEW", audit);
    }

    public void event(Job job, String type, Map<String, Object> extra) {
        var payload = json.object(extra).put("jobId", job.id()).put("revision", job.revision())
                .put("status", job.status());
        outbox.enqueue("canvas-text:" + job.id() + ":" + job.revision() + ":" + type, type, job.id(), payload);
    }

    private Job read(CanvasTextJobEntity r) {
        String result = r.getResultJson();
        return new Job(r.getId(), r.getNodeRunId(), r.getRunId(),
                r.getProjectId(),
                json.read(r.getInputJson(), Input.class), r.getStatus(), r.getPhase(),
                r.getCancelRequested(), r.getPermitHeld(), time(r.getRequestStartedAt()),
                result == null ? null : json.read(result, ObjectNode.class), r.getResultAccepted(),
                r.getRevision(), r.getErrorMsg(),
                r.getOwnerInstanceId(), r.getOwnerEpoch(), time(r.getLeaseUntil()),
                time(r.getDeadlineAt()));
    }

    private Instant time(LocalDateTime value) {
        return value == null ? null : value.toInstant(ZoneOffset.UTC);
    }
}
