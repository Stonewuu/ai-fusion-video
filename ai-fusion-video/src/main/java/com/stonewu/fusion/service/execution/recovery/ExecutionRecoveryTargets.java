package com.stonewu.fusion.service.execution.recovery;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.stonewu.fusion.entity.generation.GenerationAttemptEntity;
import com.stonewu.fusion.mapper.ai.ComfyUiTestRunMapper;
import com.stonewu.fusion.mapper.canvas.CanvasMediaJobMapper;
import com.stonewu.fusion.mapper.canvas.CanvasNodeRunMapper;
import com.stonewu.fusion.mapper.execution.ExecutionRecoveryMapper;
import com.stonewu.fusion.mapper.execution.ExecutionRecoveryMapper.Observation;
import com.stonewu.fusion.mapper.generation.GenerationAttemptMapper;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.mapper.media.EpisodeAudioJobMapper;
import com.stonewu.fusion.mapper.media.MediaJobSubmissionMapper;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import com.stonewu.fusion.service.media.job.EpisodeAudioJobs;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.util.List;

import static com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.*;

@Repository
@RequiredArgsConstructor
public class ExecutionRecoveryTargets {
    private final ExecutionRecoveryMapper recovery;
    private final GenerationAttemptMapper attempts;
    private final GenerationSubmissionMapper submissions;
    private final CanvasNodeRunMapper nodes;
    private final CanvasMediaJobMapper canvasMedia;
    private final ComfyUiTestRunMapper comfy;
    private final EpisodeAudioJobMapper audioRows;
    private final MediaJobSubmissionMapper mediaSubmissions;
    private final GenerationJobRepository generations;
    private final MediaJobRepository media;
    private final EpisodeAudioJobs audio;

    public List<Ref> candidates(int page) {
        return recovery.candidates(page * 50);
    }

    public Target get(Ref ref) {
        if (ref.kind() == Kind.AGENT_MODEL) {
            return observed(ref.kind(), recovery.agentModel(ref.id()), "智能体模型调用");
        }
        if (ref.kind() == Kind.GENERATION) {
            var job = generations.get(ref.id());
            var submission = generations.submission(job.submissionId());
            var remote = attempts.selectList(Wrappers.<GenerationAttemptEntity>lambdaQuery()
                            .eq(GenerationAttemptEntity::getJobId, job.id()).isNotNull(GenerationAttemptEntity::getSubmittedAt)
                            .orderByAsc(GenerationAttemptEntity::getAttemptNo)).stream()
                    .map(row -> new Remote("生成", row.getId(), row.getProviderTaskId(),
                            row.getAcknowledgedAt() != null,
                            row.getResultJson() != null)).toList();
            return new Target(ref.kind(), job.id(), job.state(), job.revision(), submission.context().actor(),
                    submission.projectId(), submission.context().canvasId(), runForNode(submission.canvasNodeRunId()),
                    job.permitHeld(), job.cancelRequested(), remote);
        }
        if (ref.kind() == Kind.MCP) {
            return observed(ref.kind(), recovery.mcp(ref.id()), "MCP 工具调用");
        }
        if (ref.kind() == Kind.TEXT) {
            return observed(ref.kind(), recovery.text(ref.id()), "文本生成");
        }
        var job = media.get(ref.id());
        String expectedKind = switch (ref.kind()) {
            case CANVAS_MEDIA -> "CANVAS_MEDIA";
            case COMFY_TEST -> "COMFY_WORKFLOW_TEST";
            case EPISODE_AUDIO -> "EPISODE_COMPOSE";
            default -> throw missing();
        };
        if (!job.kind().equals(expectedKind)) {
            throw missing();
        }
        List<Remote> remote;
        boolean held;
        String run = null;
        if (ref.kind() == Kind.COMFY_TEST) {
            var row = comfy.selectById(ref.id());
            if (row == null) {
                throw missing();
            }
            held = row.getPermitHeld();
            remote = row.getSubmittedAt() == null ? List.of() : List.of(new Remote("工作流试运行", job.id(),
                    row.getPromptId(), row.getAcknowledgedAt() != null, row.getResultJson() != null));
        } else if (ref.kind() == Kind.CANVAS_MEDIA) {
            var row = canvasMedia.selectById(ref.id());
            if (row == null) {
                throw missing();
            }
            held = row.getPermitHeld();
            run = runForNode(row.getNodeRunId());
            remote = media.submission(job.id(), "audio").stream().map(value -> new Remote("音频", value.attempt(),
                    value.providerId(), value.providerId() != null, row.getResultJson() != null)).toList();
        } else {
            var row = audioRows.selectById(ref.id());
            if (row == null) {
                throw missing();
            }
            held = row.getPermitHeld();
            remote = List.of("music", "sfx").stream().flatMap(step -> media.submission(job.id(), step).stream()
                    .map(value -> new Remote(step.equals("music") ? "配乐" : "音效", value.attempt(),
                            value.providerId(),
                            value.providerId() != null, audio.result(job.id(), step) != null))).toList();
        }
        return new Target(ref.kind(), job.id(), job.status(), job.revision(), job.context().actor(), job.projectId(),
                job.context().canvasId(), run, held, job.cancelRequested(), remote);
    }

    /**
     * 先锁管理员与原用户，再按画布→运行→项目→任务→子执行的顺序锁定。
     */
    public Target lock(Target before) {
        if (before.canvasId() != null) {
            recovery.lockCanvas(before.canvasId());
        }
        if (before.runId() != null) {
            recovery.lockRun(before.runId());
        }
        if (before.projectId() != null) {
            recovery.lockProject(before.projectId());
        }
        if (before.kind() == Kind.GENERATION) {
            var submission = generations.submission(generations.get(before.id()).submissionId());
            recovery.lockGenerationTask(submission.kind(), submission.taskId());
            submissions.lock(submission.id());
            recovery.lockExecution(before.kind(), before.id());
            attempts.lockHistory(before.id());
        } else {
            if (before.kind() == Kind.COMFY_TEST) {
                recovery.lockWorkflow(comfy.selectById(before.id()).getWorkflowId());
            }
            recovery.lockExecution(before.kind(), before.id());
            if (before.kind() == Kind.COMFY_TEST) {
                comfy.lock(before.id());
            } else if (before.kind() == Kind.CANVAS_MEDIA || before.kind() == Kind.EPISODE_AUDIO) {
                mediaSubmissions.lockAttempts(before.id());
            }
        }
        return get(new Ref(before.kind(), before.id()));
    }

    private Target observed(Kind kind, Observation row, String step) {
        if (row == null) {
            throw missing();
        }
        var remote = row.started() ? List.of(
                new Remote(step, row.id(), null, row.received(), row.received())) : List.<Remote>of();
        return new Target(kind, row.id(), row.status(), row.revision(),
                new CanvasActor(row.actorUserId(), row.actorTeamId()),
                row.projectId(), row.canvasId(), row.runId(), row.permitHeld(), row.cancelled(), remote);
    }

    private String runForNode(String id) {
        if (id == null) {
            return null;
        }
        var node = nodes.selectById(id);
        if (node == null) {
            throw missing();
        }
        return node.getRunId();
    }

    private static MediaException missing() {
        return new MediaException(404, "RECOVERY_TASK_NOT_FOUND", "待核实执行不存在或类型不符");
    }
}
