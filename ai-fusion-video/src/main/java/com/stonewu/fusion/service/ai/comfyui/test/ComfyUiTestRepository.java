package com.stonewu.fusion.service.ai.comfyui.test;

import com.stonewu.fusion.mapper.ai.ComfyUiTestRunMapper;
import com.stonewu.fusion.entity.ai.ComfyUiTestRunEntity;
import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.mapper.ai.ComfyUiWorkflowMapper;
import com.stonewu.fusion.mapper.ai.ComfyUiWorkflowVersionMapper;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiPreparedSubmission;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiStoredOutput;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiJobResult;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobFence;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;
import java.util.Optional;
import java.util.Set;

@Repository
@RequiredArgsConstructor
public class ComfyUiTestRepository {
    public static final String KIND = "COMFY_WORKFLOW_TEST";
    private final ComfyUiTestRunMapper testRuns;
    private final MediaJobMapper mediaJobs;
    private final UserMapper users;
    private final ComfyUiWorkflowMapper workflows;
    private final ComfyUiWorkflowVersionMapper versions;
    private final ExecutionCoordinationMapper coordination;
    private final CanvasJson json;
    private final MediaJobRepository jobs;
    private final MediaJobFence fence;
    private final GenerationOutbox outbox;

    public ComfyUiTestRun get(String id) {
        var job = jobs.get(id);
        if (!KIND.equals(job.kind())) {
            throw new MediaException(404, "COMFY_TEST_NOT_FOUND", "工作流试运行不存在");
        }
        var row = testRuns.selectById(id);
        if (row == null) {
            throw new MediaException(404, "COMFY_TEST_NOT_FOUND", "工作流试运行不存在");
        }
        return new ComfyUiTestRun(job, row.getWorkflowId(), row.getVersionId(),
                row.getWorkflowHash(),
                row.getApiConfigId(), row.getApiConfigHash(), row.getPromptId(),
                row.getRequestJson() == null ? null : json.read(row.getRequestJson(),
                        ObjectNode.class),
                row.getSubmittedAt(),
                row.getAcknowledgedAt(),
                row.getResultJson() == null ? null : json.read(row.getResultJson(),
                        ComfyUiJobResult.class),
                row.getPermitHeld(), row.getCancelSent(), row.getPollAttempts(),
                row.getConsecutiveErrors());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Optional<ComfyUiTestRun> claim(String owner) {
        coordination.lockGenerationDispatch();
        var due = mediaJobs.dueComfyTests();
        for (String id : due) {
            var run = get(id);
            if (!run.permitHeld() && run.submittedAt() == null && !run.job().cancelRequested()
                    && run.job().deadline().isAfter(jobs.now())) {
                long active = testRuns.selectCount(Wrappers.<ComfyUiTestRunEntity>lambdaQuery()
                        .eq(ComfyUiTestRunEntity::getApiConfigId, run.apiConfigId())
                        .eq(ComfyUiTestRunEntity::getPermitHeld, true));
                if (active >= 1) {
                    continue;
                }
                testRuns.update(Wrappers.<ComfyUiTestRunEntity>lambdaUpdate().eq(ComfyUiTestRunEntity::getJobId, id)
                        .set(ComfyUiTestRunEntity::getPermitHeld, true));
            }
            mediaJobs.claimCompose(owner, id);
            return Optional.of(get(id));
        }
        return Optional.empty();
    }

    public boolean heartbeat(MediaJob.Claim claim) {
        return mediaJobs.heartbeatComfyTest(claim.id(), claim.owner(), claim.epoch()) == 1;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void begin(MediaJob.Claim claim, ComfyUiPreparedSubmission prepared) {
        var run = get(claim.id());
        fence.require(claim, run.job().context(), false);
        if (run.submittedAt() != null) {
            throw MediaException.invalid("试运行已有提交意图，必须先查询原任务");
        }
        testRuns.begin(prepared.promptId(), json.write(prepared.prompt()), claim.id());
        mediaJobs.submittingComfyTest(claim.id());
        changed(claim.id(), "COMFY_TEST_CHANGED", Map.of());
    }

    /**
     * 租约交接后也只记录受理事实，不因此继续操作或写回工作流。
     */
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void acknowledge(String id, String promptId) {
        var rows = testRuns.lockUnacknowledged(id, promptId);
        if (rows.isEmpty()) {
            return;
        }
        testRuns.acknowledge(id);
        var job = jobs.get(id);
        outbox.enqueue("comfy:" + id + ":receipt", "COMFY_TEST_RECEIPT", id,
                Map.of("jobId", id, "revision", job.revision(), "promptId", promptId));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void received(MediaJob.Claim claim, ComfyUiJobResult result) {
        lockLive(claim);
        if (!result.completed()) {
            throw MediaException.invalid("只能保存已确认完成的远端结果");
        }
        testRuns.update(Wrappers.<ComfyUiTestRunEntity>lambdaUpdate()
                .eq(ComfyUiTestRunEntity::getJobId, claim.id())
                .set(ComfyUiTestRunEntity::getResultJson, json.write(result))
                .set(ComfyUiTestRunEntity::getPermitHeld, false)
                .set(ComfyUiTestRunEntity::getConsecutiveErrors, 0)
                .setIncrBy(ComfyUiTestRunEntity::getPollAttempts, 1));
        mediaJobs.ingestingComfyTest(claim.id());
        changed(claim.id(), "COMFY_TEST_CHANGED", Map.of());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void waiting(MediaJob.Claim claim, String phase, int seconds, String error) {
        lockLive(claim);
        testRuns.polled(error, claim.id());
        mediaJobs.waitForComfyTest(phase, error, Math.max(1, Math.min(60, seconds)), claim.id());
        changed(claim.id(), "COMFY_TEST_CHANGED", Map.of());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public boolean markCancelSent(MediaJob.Claim claim) {
        lockLive(claim);
        return testRuns.update(
                Wrappers.<ComfyUiTestRunEntity>lambdaUpdate().eq(ComfyUiTestRunEntity::getJobId, claim.id())
                        .eq(ComfyUiTestRunEntity::getCancelSent, false)
                        .set(ComfyUiTestRunEntity::getCancelSent, true)) == 1;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void output(MediaJob.Claim claim, int index, ComfyUiStoredOutput output) {
        var job = jobs.check(claim);
        var checkpoint = job.checkpoint().deepCopy();
        var outputs = checkpoint.withArray("outputs");
        if (index < outputs.size()) {
            return;
        }
        if (index != outputs.size()) {
            throw MediaException.invalid("试运行结果保存顺序无效");
        }
        outputs.add(json.object(output));
        mediaJobs.importPlacement(json.write(checkpoint), claim.id());
        changed(claim.id(), "COMFY_TEST_CHANGED", Map.of());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public void finish(MediaJob.Claim claim, String status, String error, boolean keepPermit) {
        if (!Set.of("SUCCEEDED", "FAILED", "CANCELLED", "NEEDS_REVIEW").contains(status)) {
            throw MediaException.invalid("试运行状态无效");
        }
        var run = get(claim.id());
        users.lock(run.job().context().actor().userId());
        workflows.lock(run.workflowId());
        lockLive(claim);
        if (status.equals("SUCCEEDED")) {
            fence.require(claim, run.job().context(), false);
        }
        testRuns.update(Wrappers.<ComfyUiTestRunEntity>lambdaUpdate().eq(ComfyUiTestRunEntity::getJobId, claim.id())
                .set(ComfyUiTestRunEntity::getPermitHeld, keepPermit));
        mediaJobs.finishComfyTest(status, error, claim.id());
        recordVersion(get(claim.id()));
        changed(claim.id(), "COMFY_TEST_CHANGED", Map.of());
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public void recordVersion(ComfyUiTestRun run) {
        int status = switch (run.job().status()) {
            case "SUCCEEDED" -> 1;
            case "FAILED" -> 2;
            case "CANCELLED" -> 4;
            case "NEEDS_REVIEW" -> 5;
            default -> 3;
        };
        String message = run.job().error() != null ? run.job().error() : switch (status) {
            case 1 -> "试运行成功，输出已保存";
            case 4 -> "试运行已取消";
            default -> "试运行任务已受理，可关闭页面后继续查看";
        };
        var snapshot = json.read(json.write(run.job().input()), ComfyUiTestSnapshot.class);
        versions.recordTest(status, message, status == 1 ? run.apiConfigHash() : null, run.versionId(), run.job().id(),
                run.workflowHash(), snapshot.workflow().getModelType(), run.apiConfigId());
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public void changed(String id, String type, Map<String, Object> detail) {
        var job = jobs.get(id);
        var payload = json.object(
                Map.of("jobId", id, "revision", job.revision(), "status", job.status(), "phase", job.phase()));
        payload.set("detail", json.object(detail));
        outbox.enqueue("comfy:" + id + ":" + job.revision() + ":" + type, type, id, payload);
    }

    private void lockLive(MediaJob.Claim claim) {
        if (mediaJobs.lockComfyTestOwner(claim.id(), claim.owner(), claim.epoch()).size() != 1) {
            throw MediaJobFence.lost();
        }
    }
}
