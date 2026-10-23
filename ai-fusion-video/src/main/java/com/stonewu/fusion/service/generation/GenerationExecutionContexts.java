package com.stonewu.fusion.service.generation;

import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.job.GenerationExecutionIdentity;
import com.stonewu.fusion.service.generation.job.GenerationJobException;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import com.stonewu.fusion.service.generation.job.GenerationLeaseScope;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Objects;

@Component
@RequiredArgsConstructor
public class GenerationExecutionContexts {
    private final GenerationTasks tasks;
    private final AgentRunMapper agentRuns;
    private final ProjectMapper projects;
    private final UserMapper users;
    private final CanvasJson json;
    private final CanvasAccessPolicy access;
    private final MediaAccessPolicy mediaAccess;
    private final ResourceAgentFence agentFence;
    private final GenerationJobRepository generationJobs;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void prepare(ImageTask task, ResourceCommandContext context) {
        Long projectId = project(task.getProjectId(), context);
        agentFence.requireActive(context, projectId);
        var owner = mediaAccess.creation(projectId, context);
        task.setUserId(context.actor().userId());
        task.setProjectId(projectId);
        task.setOwnerType(owner.type());
        task.setOwnerId(owner.id());
        task.setExecutionContextJson(json.write(context));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void prepare(VideoTask task, ResourceCommandContext context) {
        Long projectId = project(task.getProjectId(), context);
        agentFence.requireActive(context, projectId);
        var owner = mediaAccess.creation(projectId, context);
        task.setUserId(context.actor().userId());
        task.setProjectId(projectId);
        task.setOwnerType(owner.type());
        task.setOwnerId(owner.id());
        task.setExecutionContextJson(json.write(context));
    }

    /**
     * 私有上下文不通过任务缓存/HTTP DTO 传输，每次从持久化任务读取。
     */
    public Snapshot get(GenerationKind kind, long taskId) {
        var row = tasks.context(kind, taskId);
        if (row == null) {
            throw new MediaException(404, "GENERATION_NOT_FOUND", "生成任务不存在");
        }
        String stored = row.executionContextJson();
        if (stored == null || stored.isBlank()) {
            throw new MediaException(409, "GENERATION_CONTEXT_MISSING", "此任务缺少执行上下文，请重新提交生成");
        }
        var context = json.read(stored, ResourceCommandContext.class);
        if (context.actor().userId() != row.userId()) {
            throw new IllegalStateException("生成任务身份与持久化上下文不符");
        }
        return new Snapshot(kind, taskId, row.projectId(), row.status(), context);
    }

    /**
     * 调用者先取得 Agent/用户行锁，本方法按项目、任务顺序获取后续锁。
     */
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Snapshot running(GenerationKind kind, long taskId) {
        var snapshot = get(kind, taskId);
        agentFence.requireActive(snapshot.context(), snapshot.projectId());
        users.lockEnabled(snapshot.context().actor().userId());
        return lockActive(kind, taskId, snapshot.context());
    }

    /**
     * 调用者先取得 Agent/用户行锁，本方法按项目、任务顺序获取后续锁。
     */
    @Transactional(propagation = Propagation.MANDATORY)
    public Snapshot lockActive(GenerationKind kind, long taskId, ResourceCommandContext expected) {
        var before = get(kind, taskId);
        if (before.projectId() != null) {
            projects.lock(before.projectId());
        }
        tasks.lock(kind, taskId);
        var current = get(kind, taskId);
        if (current.status() != 1 || !current.context().equals(expected) || !Objects.equals(current.projectId(),
                before.projectId())) {
            throw new MediaException(409, "GENERATION_FENCED", "生成任务已停止或执行身份已变化");
        }
        agentFence.requireActive(expected, current.projectId());
        mediaAccess.creation(current.projectId(), expected);
        if ("system".equals(expected.origin())) {
            var job = generationJobs.leased(GenerationLeaseScope.require(), false);
            var submission = generationJobs.submission(job.submissionId());
            if (submission.kind() != kind || submission.taskId() != taskId
                    || !Objects.equals(submission.projectId(), current.projectId())
                    || !GenerationExecutionIdentity.worker(submission.context()).equals(expected)) {
                throw GenerationJobException.fenced();
            }
        }
        return current;
    }

    public List<Long> readableProjects(CanvasActor actor) {
        return access.readableProjects(actor);
    }

    public void requireRead(GenerationKind kind, long taskId, CanvasActor actor) {
        access.requireActorScope(actor);
        var scope = tasks.context(kind, taskId);
        if (scope == null) {
            throw new MediaException(404, "GENERATION_NOT_FOUND", "生成任务不存在");
        }
        if (scope.projectId() != null) {
            access.requireProject(scope.projectId(), actor);
            return;
        }
        if (Objects.equals(scope.ownerType(), 1) && Objects.equals(scope.ownerId(), actor.userId())) {
            return;
        }
        if (Objects.equals(scope.ownerType(), 2) && actor.teamId() != null && Objects.equals(scope.ownerId(),
                actor.teamId())) {
            return;
        }
        throw new MediaException(403, "GENERATION_ACCESS_DENIED", "无权读取此生成任务");
    }

    private Long project(Long requested, ResourceCommandContext context) {
        if (!"agent".equals(context.origin())) {
            return requested;
        }
        var agent = agentRuns.selectOne(
                new LambdaQueryWrapper<AgentRun>().select(AgentRun::getRunId, AgentRun::getProjectId)
                        .eq(AgentRun::getRunId, context.agentRunId())
                        .eq(AgentRun::getUserId, context.actor().userId()));
        if (agent == null) {
            throw new MediaException(403, "GENERATION_ACCESS_DENIED", "智能体运行身份不可用");
        }
        Long inherited = agent.getProjectId();
        if (inherited != null && requested != null && !inherited.equals(requested)) {
            throw new MediaException(403, "GENERATION_SCOPE_MISMATCH", "生成任务不能跨越智能体项目范围");
        }
        return inherited == null ? requested : inherited;
    }

    public record Snapshot(GenerationKind kind, long taskId, Long projectId, int status,
                           ResourceCommandContext context) {
    }

}
