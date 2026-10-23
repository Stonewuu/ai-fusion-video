package com.stonewu.fusion.service.execution.recovery;

import com.stonewu.fusion.mapper.ai.AgentModelCallUsageMapper;
import com.stonewu.fusion.mapper.ai.AgentModelRequestMapper;
import com.stonewu.fusion.mapper.canvas.CanvasTextJobMapper;
import com.stonewu.fusion.mapper.canvas.CanvasMcpJobMapper;
import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.ai.ComfyUiTestRunMapper;
import com.stonewu.fusion.mapper.canvas.CanvasMediaJobMapper;
import com.stonewu.fusion.mapper.media.EpisodeAudioJobMapper;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.mapper.generation.GenerationAttemptMapper;
import com.stonewu.fusion.mapper.generation.RemoteGenerationJobMapper;
import com.stonewu.fusion.mapper.generation.GenerationItemStateMapper;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.stonewu.fusion.entity.ai.ComfyUiTestRunEntity;
import com.stonewu.fusion.entity.canvas.CanvasMediaJobEntity;
import com.stonewu.fusion.entity.canvas.CanvasTextJobEntity;
import com.stonewu.fusion.entity.media.EpisodeAudioJobEntity;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;

import com.stonewu.fusion.service.ai.comfyui.test.ComfyUiTestRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.run.text.CanvasTextRepository;
import com.stonewu.fusion.service.generation.job.GenerationJobLifecycle;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.media.job.EpisodeAudioJobs;
import com.stonewu.fusion.service.media.job.MediaJobNotifications;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;

import static com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Kind;
import static com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Target;

/**
 * 仅由核实服务在权限、版本、远端结束确认及锁均已取得后调用。不得发布输出或创建新的供应商请求。
 */
@Component
@RequiredArgsConstructor
public class ExecutionRecoveryCloser {
    private final AgentModelCallUsageMapper modelUsage;
    private final AgentModelRequestMapper modelRequests;
    private final CanvasTextJobMapper textJobs;
    private final CanvasMcpJobMapper mcpJobs;
    private final MediaJobMapper mediaJobs;
    private final ComfyUiTestRunMapper comfyRuns;
    private final CanvasMediaJobMapper canvasMedia;
    private final EpisodeAudioJobMapper audioJobs;
    private final GenerationSubmissionMapper submissions;
    private final GenerationAttemptMapper attempts;
    private final RemoteGenerationJobMapper remoteJobs;
    private final GenerationItemStateMapper items;
    private final CanvasJson json;
    private final GenerationJobRepository generations;
    private final GenerationJobLifecycle generationLifecycle;
    private final GenerationOutbox outbox;
    private final CanvasTextRepository texts;
    private final MediaJobRepository media;
    private final MediaJobNotifications notifications;
    private final ComfyUiTestRepository comfy;
    private final EpisodeAudioJobs audio;

    @Transactional(propagation = Propagation.MANDATORY)
    public void close(Target target, long admin, String decisionId, String note) {
        var audit = Map.<String, Object>of("action", "ADMIN_CONFIRM_FINISHED", "adminUserId", admin, "decisionId",
                decisionId, "note", note);
        String message = "管理员已核实远端调用结束，原身份的后续执行已停止";
        if (target.kind() == Kind.AGENT_MODEL) {
            modelUsage.adminCancel(target.id());
            modelRequests.adminCancel(target.id());
            return;
        }
        if (target.kind() == Kind.GENERATION) {
            generation(target.id(), admin, note, decisionId, audit, message);
            return;
        }
        if (target.kind() == Kind.TEXT) {
            textJobs.update(Wrappers.<CanvasTextJobEntity>lambdaUpdate()
                    .eq(CanvasTextJobEntity::getId, target.id())
                    .set(CanvasTextJobEntity::getCancelRequested, true)
                    .setIncrBy(CanvasTextJobEntity::getOwnerEpoch, 1));
            texts.finish(texts.get(target.id(), false), "CANCELLED", message, false);
            texts.event(texts.get(target.id(), false), "CANVAS_TEXT_REVIEW", audit);
            return;
        }
        if (target.kind() == Kind.MCP) {
            mcpJobs.adminCancel(message, target.id());
            return;
        }
        mediaJobs.adminCancel(message, target.id());
        if (target.kind() == Kind.COMFY_TEST) {
            comfyRuns.update(
                    Wrappers.<ComfyUiTestRunEntity>lambdaUpdate().eq(ComfyUiTestRunEntity::getJobId, target.id())
                            .set(ComfyUiTestRunEntity::getPermitHeld, false));
            comfy.recordVersion(comfy.get(target.id()));
            comfy.changed(target.id(), "COMFY_TEST_REVIEW", audit);
        } else {
            if (target.kind() == Kind.CANVAS_MEDIA) {
                canvasMedia.update(
                        Wrappers.<CanvasMediaJobEntity>lambdaUpdate().eq(CanvasMediaJobEntity::getJobId, target.id())
                                .set(CanvasMediaJobEntity::getPermitHeld, false));
            } else {
                var evidence = json.object(audit).put("recordedAt", media.now().toString());
                audioJobs.update(
                        Wrappers.<EpisodeAudioJobEntity>lambdaUpdate().eq(EpisodeAudioJobEntity::getJobId, target.id())
                                .set(EpisodeAudioJobEntity::getPermitHeld, false)
                                .set(EpisodeAudioJobEntity::getResolvedStepsJson,
                                        json.write(audio.unresolved(target.id())))
                                .set(EpisodeAudioJobEntity::getReviewJson, json.write(evidence)));
                notifications.finished(media.get(target.id()));
            }
            var payload = json.object(audit).put("jobId", target.id())
                    .put("revision", media.get(target.id()).revision()).put("status", "CANCELLED");
            outbox.enqueue("admin-recovery:" + decisionId,
                    target.kind() == Kind.CANVAS_MEDIA ? "CANVAS_MEDIA_REVIEW" : "EPISODE_AUDIO_REVIEW", target.id(),
                    payload);
        }
    }

    private void generation(String id, long admin, String note, String decisionId, Map<String, Object> audit,
                            String message) {
        var job = generations.get(id);
        var submission = generations.submission(job.submissionId());
        submissions.update(Wrappers.<GenerationSubmissionEntity>lambdaUpdate()
                .eq(GenerationSubmissionEntity::getId, submission.id())
                .set(GenerationSubmissionEntity::getCancelRequested, true));
        attempts.adminResolve(note, admin, id);
        remoteJobs.adminCancel(message, id);
        for (long itemId : job.itemIds()) {
            items.failTaskItem(submission.kind(), itemId, submission.taskId(), message);
        }
        generationLifecycle.aggregate(submission.id());
        var payload = json.object(audit).put("submissionId", submission.id()).put("jobId", id)
                .put("revision", generations.submission(submission.id()).revision());
        outbox.enqueue("admin-recovery:" + decisionId, "GENERATION_CHANGED", submission.id(), payload);
    }
}
