package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class GenerationRequestCommands {
    private final GenerationRequestRepository requests;
    private final GenerationJobRepository jobs;
    private final GenerationJobCommands commands;
    private final GenerationSubmissionMapper submissions;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = {"imageTask", "videoTask", "imageItems", "videoItems"}, allEntries = true)
    public Withdrawal withdraw(String requestId, GenerationKind kind, CanvasActor actor) {
        if (kind == null) {
            throw GenerationJobException.invalid("请选择图片或视频生成类别");
        }
        var context = ResourceCommandContext.user(actor, null);
        requests.lock(requestId, context);
        var row = submissions.selectOne(new LambdaQueryWrapper<GenerationSubmissionEntity>()
                .select(GenerationSubmissionEntity::getId)
                .eq(GenerationSubmissionEntity::getScopeKey, requests.scope(context))
                .eq(GenerationSubmissionEntity::getRequestId, requestId));
        if (row == null) {
            requests.withdraw(requestId, kind, context);
            return new Withdrawal(null, null, "CANCELLED_BEFORE_ACCEPTANCE");
        }
        var submission = jobs.submission(row.getId());
        if (submission.kind() != kind) {
            throw GenerationJobException.invalid("请求标识属于其他生成类别");
        }
        commands.cancel(kind, submission.taskId(), actor);
        requests.withdraw(requestId, kind, context);
        var current = jobs.submission(submission.id());
        return new Withdrawal(current.id(), current.publicTaskId(), current.state());
    }

    public record Withdrawal(String submissionId, String taskId, String state) {
    }
}
