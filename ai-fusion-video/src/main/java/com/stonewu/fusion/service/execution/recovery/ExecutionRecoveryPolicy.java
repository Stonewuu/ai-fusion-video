package com.stonewu.fusion.service.execution.recovery;

import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.mapper.system.UserRoleMapper;
import com.stonewu.fusion.entity.system.User;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import com.stonewu.fusion.service.media.job.MediaJobFence;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;

import static com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.*;

@Component
@RequiredArgsConstructor
public class ExecutionRecoveryPolicy {
    private final UserMapper users;
    private final UserRoleMapper userRoles;
    private final CanvasAccessPolicy access;
    private final CanvasDocumentRepository canvases;
    private final GenerationJobRepository generations;
    private final GenerationExecutionContexts generationAccess;
    private final MediaJobFence mediaFence;
    private final CanvasJson json;

    public void requireAdmin(long userId, boolean lock) {
        boolean active = lock ? users.lockEnabled(userId) != null : users.exists(Wrappers.<User>lambdaQuery()
                .eq(User::getId, userId).eq(User::getStatus, 1));
        var roles = userRoles.activeAdminRoles(userId, lock);
        if (!active || roles.isEmpty()) {
            throw new MediaException(403, "RECOVERY_ADMIN_REQUIRED", "只有有效的平台管理员可以核实失效身份的任务");
        }
    }

    public boolean originalCanControl(Target target) {
        try {
            access.requireActorScope(target.actor());
            if (target.canvasId() != null) {
                if (!access.requireView(canvases.require(target.canvasId(), false), target.actor()).canExecute()) {
                    return false;
                }
            }
            switch (target.kind()) {
                case GENERATION -> {
                    var submission = generations.submission(generations.get(target.id()).submissionId());
                    generationAccess.requireRead(submission.kind(), submission.taskId(), target.actor());
                }
                case EPISODE_AUDIO -> access.requireProjectEditor(target.projectId(), target.actor());
                case COMFY_TEST -> mediaFence.requireWorkflowTester(target.actor());
                case CANVAS_MEDIA, TEXT, MCP -> {
                    if (target.canvasId() == null) {
                        return false;
                    }
                }
                case AGENT_MODEL -> {
                }
            }
            return true;
        } catch (BusinessException failure) {
            if (failure.getCode() == 403 || failure.getCode() == 404) {
                return false;
            }
            throw failure;
        }
    }

    public View view(Target target) {
        boolean modelRequest = target.kind() == Kind.AGENT_MODEL;
        boolean canResolve = (target.status().equals("NEEDS_REVIEW") || target.status().equals("UNKNOWN_SUBMISSION"))
                && (modelRequest || !originalCanControl(target));
        return new View(target.kind(), target.id(), target.status(), target.revision(), target.actor().userId(),
                target.actor().teamId(),
                target.projectId(), target.canvasId(), target.capacityHeld(), target.cancelled(), target.remote(),
                json.hash(target), canResolve,
                canResolve ? modelRequest ? "模型请求没有完整响应，管理员可核实供应商记录后释放额度" : "原账号或作用域权限已失效，可核实远端结束后收尾"
                        : "由原发起人在对应工作区核实，或任务已结束");
    }
}
