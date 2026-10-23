package com.stonewu.fusion.service.ai.comfyui.test;

import com.stonewu.fusion.entity.ai.ComfyUiTestWithdrawalEntity;

import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.ai.ComfyUiTestRunMapper;
import com.stonewu.fusion.mapper.ai.ComfyUiTestWithdrawalMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.entity.ai.ComfyUiTestRunEntity;
import com.stonewu.fusion.entity.media.MediaJobEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.mapper.ai.ComfyUiWorkflowMapper;
import com.stonewu.fusion.mapper.ai.ComfyUiWorkflowVersionMapper;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ProviderConfigurationFingerprint;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiStoredOutput;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.job.MediaJobFence;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.ZoneOffset;
import java.util.*;

@Service
@RequiredArgsConstructor
public class ComfyUiTestService {
    private final MediaJobMapper mediaJobs;
    private final ComfyUiTestRunMapper testRuns;
    private final ComfyUiTestWithdrawalMapper withdrawals;
    private final UserMapper users;
    private final CanvasJson json;
    private final ComfyUiTestRepository runs;
    private final MediaJobFence fence;
    private final ComfyUiTestInputs inputs;
    private final ComfyUiWorkflowMapper workflows;
    private final ComfyUiWorkflowVersionMapper versions;
    private final ApiConfigService apis;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public ComfyUiTestView submit(String requestId, long versionId, String workflowHash,
                                  Map<String, Object> values, CanvasActor actor) {
        lockActor(actor);
        if (values == null || values.size() > 64 || json.write(values).length() > 64 * 1024 * 1024) {
            throw MediaException.invalid("试运行输入超过限制");
        }
        String requestKey = requestKey(requestId);
        String hash = json.hash(
                Map.of("actor", actor, "versionId", versionId, "workflowHash", workflowHash, "inputs", values));
        var existing = mediaJobs.selectOne(Wrappers.<MediaJobEntity>lambdaQuery()
                .eq(MediaJobEntity::getActorUserId, actor.userId()).eq(MediaJobEntity::getRequestId, requestKey));
        if (existing != null) {
            if (!hash.equals(existing.getRequestHash())) {
                throw new MediaException(409, "COMFY_TEST_REQUEST_CONFLICT",
                        "同一试运行请求不能使用不同的版本、输入或身份");
            }
            return view(runs.get(existing.getId()));
        }
        if (withdrawals.exists(Wrappers.<ComfyUiTestWithdrawalEntity>lambdaQuery()
                .eq(ComfyUiTestWithdrawalEntity::getActorUserId, actor.userId())
                .eq(ComfyUiTestWithdrawalEntity::getRequestKey, requestKey))) {
            throw new MediaException(410, "COMFY_TEST_REQUEST_WITHDRAWN", "这次试运行请求已撤回，不能再次受理");
        }
        var observed = versions.selectById(versionId);
        if (observed == null) {
            throw new MediaException(404, "COMFY_VERSION_NOT_FOUND", "工作流版本不存在");
        }
        var workflow = workflows.selectForUpdate(observed.getWorkflowId());
        if (workflow == null || !Integer.valueOf(1).equals(workflow.getStatus())) {
            throw MediaException.invalid("工作流不存在或已禁用");
        }
        var version = versions.selectForUpdate(versionId);
        if (version == null || !Objects.equals(workflowHash, version.getWorkflowHash())) {
            throw new MediaException(409, "COMFY_TEST_VERSION_CHANGED", "工作流草稿已经变化，请刷新后重新试运行");
        }
        if (!Integer.valueOf(ComfyUiWorkflowVersion.VALIDATION_VALID).equals(version.getValidationStatus())) {
            throw MediaException.invalid("请先完成目标 ComfyUI 在线依赖校验");
        }
        var api = requireApi(workflow.getApiConfigId());
        var snapshot = inputs.freeze(workflow, version, values, actor);
        long active = mediaJobs.selectCount(
                Wrappers.<MediaJobEntity>lambdaQuery().eq(MediaJobEntity::getActorUserId, actor.userId())
                        .eq(MediaJobEntity::getKind, "COMFY_WORKFLOW_TEST")
                        .in(MediaJobEntity::getStatus, List.of("QUEUED", "WAITING", "RUNNING", "NEEDS_REVIEW")));
        if (active >= 8) {
            throw new MediaException(429, "COMFY_TEST_LIMIT", "最多同时保留 8 个活动试运行，请先处理已有任务");
        }
        String id = UUID.randomUUID().toString();
        mediaJobs.insertComfyTest(id, actor.userId(), requestKey, hash,
                json.write(ResourceCommandContext.user(actor, null)), json.write(snapshot),
                workflow.getModelType() == 2 ? 25 : 60);
        testRuns.insert(ComfyUiTestRunEntity.builder().jobId(id).workflowId(workflow.getId()).versionId(versionId)
                .workflowHash(version.getWorkflowHash()).apiConfigId(api.getId())
                .apiConfigHash(ProviderConfigurationFingerprint.of(api, json)).actorTeamId(actor.teamId()).build());
        versions.update(Wrappers.<ComfyUiWorkflowVersion>lambdaUpdate().eq(ComfyUiWorkflowVersion::getId, versionId)
                .set(ComfyUiWorkflowVersion::getLastTestRunId, id)
                .set(ComfyUiWorkflowVersion::getTestApiConfigHash, null));
        runs.recordVersion(runs.get(id));
        runs.changed(id, "COMFY_TEST_ACCEPTED", Map.of());
        return view(runs.get(id));
    }

    public ComfyUiTestView get(String id, CanvasActor actor) {
        return view(requireOwned(id, actor));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public void withdraw(String requestId, CanvasActor actor) {
        lockActor(actor);
        String key = requestKey(requestId);
        var existing = mediaJobs.lockRequest(actor.userId(), key);
        if (existing != null) {
            var run = requireOwned(existing.getId(), actor);
            cancel(run.job().id(), actor);
        }
        var prior = withdrawals.selectOne(Wrappers.<ComfyUiTestWithdrawalEntity>lambdaQuery()
                .eq(ComfyUiTestWithdrawalEntity::getActorUserId, actor.userId())
                .eq(ComfyUiTestWithdrawalEntity::getRequestKey, key));
        if (prior != null && !Objects.equals(prior.getActorTeamId(), actor.teamId())) {
            throw new MediaException(409, "COMFY_TEST_REQUEST_CONFLICT", "请求属于其他团队作用域");
        }
        if (prior == null) {
            withdrawals.recordWithdrawal(actor.userId(), key, actor.teamId());
        }
    }

    public List<ComfyUiTestView> list(long versionId, CanvasActor actor) {
        fence.requireWorkflowTester(actor);
        return testRuns.history(versionId, actor.userId(), actor.teamId()).stream().map(id -> get(id, actor))
                .toList();
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public ComfyUiTestView cancel(String id, CanvasActor actor) {
        // 取消是针对固定任务的单向、幂等请求，不要求用户追赶后台轮询版本。
        var run = lockOwned(id, null, actor);
        if (run.job().cancelRequested() || Set.of("SUCCEEDED", "FAILED", "CANCELLED").contains(run.job().status())) {
            return view(run);
        }
        boolean immediate = run.submittedAt() == null || run.result() != null;
        mediaJobs.cancelComfyTest(
                immediate ? "CANCELLED" : run.job().status().equals("NEEDS_REVIEW") ? "WAITING" : run.job().status(),
                immediate ? "已取消试运行" : "已请求取消，等待远端确认终态", id);
        if (immediate) {
            testRuns.update(Wrappers.<ComfyUiTestRunEntity>lambdaUpdate().eq(ComfyUiTestRunEntity::getJobId, id)
                    .set(ComfyUiTestRunEntity::getPermitHeld, false));
        }
        runs.recordVersion(runs.get(id));
        runs.changed(id, "COMFY_TEST_CHANGED", Map.of("action", "CANCEL", "actorUserId", actor.userId()));
        return view(runs.get(id));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public ComfyUiTestView review(String id, long expectedRevision, String action, String note, CanvasActor actor) {
        var run = lockOwned(id, expectedRevision, actor);
        if (!actions(run).contains(action)) {
            throw new MediaException(409, "COMFY_TEST_ACTION_CHANGED", "任务状态已变化，此操作不再可用");
        }
        if (note == null || note.isBlank() || note.length() > 2000) {
            throw MediaException.invalid("请填写不超过 2000 字的核实说明");
        }
        if (action.equals("CONFIRM_FINISHED")) {
            mediaJobs.reviewComfyTest(run.job().cancelRequested() ? "CANCELLED" : "FAILED",
                    "人工确认远端已结束：" + note.trim(), id);
            testRuns.update(Wrappers.<ComfyUiTestRunEntity>lambdaUpdate().eq(ComfyUiTestRunEntity::getJobId, id)
                    .set(ComfyUiTestRunEntity::getPermitHeld, false));
        } else {
            requireApi(run.apiConfigId(), run.apiConfigHash());
            mediaJobs.resumeComfyTest(id);
            testRuns.update(Wrappers.<ComfyUiTestRunEntity>lambdaUpdate().eq(ComfyUiTestRunEntity::getJobId, id)
                    .set(ComfyUiTestRunEntity::getConsecutiveErrors, 0)
                    .set(ComfyUiTestRunEntity::getCancelSent, false));
        }
        runs.recordVersion(runs.get(id));
        runs.changed(id, "COMFY_TEST_REVIEW",
                Map.of("action", action, "note", note.trim(), "actorUserId", actor.userId()));
        return view(runs.get(id));
    }

    public ApiConfig requireApi(long id, String expectedHash) {
        var api = requireApi(id);
        if (!ProviderConfigurationFingerprint.of(api, json).equals(expectedHash)) {
            throw new MediaException(409, "COMFY_TEST_PROVIDER_CHANGED", "供应商连接已变化，请恢复原连接后核实原任务");
        }
        return api;
    }

    private ApiConfig requireApi(long id) {
        var api = apis.getById(id);
        if (api == null || !Integer.valueOf(1).equals(api.getStatus()) || !"comfyui".equalsIgnoreCase(
                api.getPlatform())) {
            throw new MediaException(409, "COMFY_TEST_PROVIDER_UNAVAILABLE", "ComfyUI 配置不存在、已停用或类型不符");
        }
        return api;
    }

    private ComfyUiTestRun requireOwned(String id, CanvasActor actor) {
        fence.requireWorkflowTester(actor);
        var run = runs.get(id);
        if (!run.job().context().actor().equals(actor)) {
            throw new MediaException(403, "COMFY_TEST_ACCESS_DENIED", "此试运行不属于当前用户和团队作用域");
        }
        return run;
    }

    private void lockActor(CanvasActor actor) {
        users.lock(actor.userId());
        fence.requireWorkflowTester(actor);
    }

    private String requestKey(String id) {
        if (id == null || !id.matches("[A-Za-z0-9_.:-]{1,96}")) {
            throw MediaException.invalid("试运行请求标识无效");
        }
        return "comfy-test:" + id;
    }

    private ComfyUiTestRun lockOwned(String id, Long expectedRevision, CanvasActor actor) {
        lockActor(actor);
        var run = requireOwned(id, actor);
        workflows.lock(run.workflowId());
        mediaJobs.lock(id);
        run = runs.get(id);
        if (expectedRevision != null && run.job().revision() != expectedRevision) {
            throw new MediaException(409, "COMFY_TEST_REVISION_CHANGED", "试运行状态已变化，请刷新后操作");
        }
        return run;
    }

    private List<String> actions(ComfyUiTestRun run) {
        if (!run.job().status().equals("NEEDS_REVIEW")) {
            return List.of();
        }
        List<String> actions = new ArrayList<>();
        if (run.promptId() != null && run.result() == null) {
            actions.add("RECHECK");
        }
        if (run.result() != null && !run.job().cancelRequested()) {
            actions.add("RETRY_INGEST");
        }
        actions.add("CONFIRM_FINISHED");
        return List.copyOf(actions);
    }

    private ComfyUiTestView view(ComfyUiTestRun run) {
        var job = run.job();
        List<ComfyUiStoredOutput> outputs = new ArrayList<>();
        job.checkpoint().path("outputs")
                .forEach(node -> outputs.add(json.read(node.toString(), ComfyUiStoredOutput.class)));
        Map<String, String> summary = new LinkedHashMap<>();
        job.input().path("inputs").fields().forEachRemaining(entry -> {
            var value = entry.getValue();
            String text = value.isArray() ? value.size() + " 项输入" : value.asText();
            if (text.startsWith("data:")) {
                text = "Data URI 媒体输入";
            }
            summary.put(entry.getKey(), text.length() > 240 ? text.substring(0, 240) + "…" : text);
        });
        return new ComfyUiTestView(job.id(), run.versionId(), run.workflowHash(), job.status(), job.phase(),
                job.error(), run.promptId(),
                run.acknowledgedAt() != null, job.cancelRequested(), run.permitHeld(), job.revision(),
                job.createdAt().toInstant(ZoneOffset.UTC), job.deadline().toInstant(ZoneOffset.UTC),
                List.copyOf(outputs), Map.copyOf(summary), actions(run));
    }
}
