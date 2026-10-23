package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.common.BusinessException;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.GenerationTasks;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.mapper.team.TeamMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;


@Component
@RequiredArgsConstructor
public class GenerationDeletionGuard {
    private final GenerationSubmissionMapper submissions;
    private final GenerationTasks tasks;
    private final GenerationExecutionContexts access;
    private final UserMapper users;
    private final TeamMapper teams;
    private final ProjectMapper projects;

    /** 在任务行锁之前锁定授权范围，与账户禁用、成员移除及项目权限变化串行化。 */
    @Transactional(propagation = Propagation.MANDATORY)
    public void lockScope(GenerationKind kind, long taskId, CanvasActor actor) {
        var task = tasks.context(kind, taskId);
        if (task == null || task.userId() != actor.userId()) {
            throw new BusinessException(404, "生成任务不存在");
        }
        users.lockEnabled(actor.userId());
        if (actor.teamId() != null) {
            teams.lockEnabled(actor.teamId());
        }
        if (task.projectId() != null) {
            projects.lock(task.projectId());
        }
        access.requireRead(kind, taskId, actor);
    }

    /**
     * 调用者持有业务任务行锁，禁止删除未确认远端结束或被运行引用的任务。
     */
    @Transactional(propagation = Propagation.MANDATORY)
    public void requireTerminal(GenerationKind kind, long taskId) {
        long references = submissions.selectCount(new LambdaQueryWrapper<GenerationSubmissionEntity>()
                .eq(GenerationSubmissionEntity::getKind, kind.name()).eq(GenerationSubmissionEntity::getTaskId, taskId)
                .isNotNull(GenerationSubmissionEntity::getCanvasNodeRunId));
        if (references > 0) {
            throw new BusinessException(409, "画布运行引用了此任务，不能从创作记录单独删除");
        }
        long active = submissions.countActiveJobs(kind.name(), taskId);
        if (active > 0) {
            throw new BusinessException(409, "供应商执行仍未确认结束，请先处理任务状态");
        }
    }
}
