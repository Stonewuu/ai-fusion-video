package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import org.springframework.stereotype.Component;

/**
 * 已受理的生成由自己的租约负责；保留父 Agent 来源，但不借用已释放的对话工作者租约。
 */
@Component
@RequiredArgsConstructor
public class GenerationExecutionIdentity {
    private final AgentRunMapper runs;
    private final MediaAccessPolicy access;

    public static ResourceCommandContext worker(ResourceCommandContext original) {
        return new ResourceCommandContext(original.actor(), "system", original.canvasId(),
                original.agentRunId(), original.toolCallId(), null, null);
    }

    public boolean permitted(GenerationSubmission submission) {
        try {
            access.creation(submission.projectId(), worker(submission.context()));
        } catch (BusinessException denied) {
            if (denied.getCode() == 403 || denied.getCode() == 404 || denied.getCode() == 410) {
                return false;
            }
            throw denied;
        }
        if (submission.context().agentRunId() == null) {
            return true;
        }
        var query = new LambdaQueryWrapper<AgentRun>()
                .eq(AgentRun::getRunId, submission.context().agentRunId())
                .eq(AgentRun::getUserId, submission.userId())
                .isNull(AgentRun::getCancelRequestedAt)
                .notIn(AgentRun::getStatus, "CANCELLED", "FAILED", "EXPIRED")
                .and(scope -> scope.isNull(AgentRun::getProjectId).or()
                        .eq(AgentRun::getProjectId, submission.projectId()));
        if (submission.teamId() == null) {
            query.isNull(AgentRun::getActorTeamId);
        } else {
            query.eq(AgentRun::getActorTeamId, submission.teamId());
        }
        return runs.selectCount(query) == 1;
    }
}
