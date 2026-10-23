package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.mapper.generation.GenerationWithdrawalMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Repository
@RequiredArgsConstructor
public class GenerationRequestRepository {
    private final GenerationWithdrawalMapper withdrawals;
    private final UserMapper users;
    private final CanvasJson json;

    public String scope(ResourceCommandContext context) {
        return json.hash(List.of(context.actor().userId(),
                context.actor().teamId() == null ? "personal" : context.actor().teamId(),
                context.origin(), context.canvasId() == null ? "generation" : context.canvasId()));
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public void lock(String requestId, ResourceCommandContext context) {
        if (requestId == null || !requestId.matches("[a-zA-Z0-9_.:-]{1,128}")) {
            throw GenerationJobException.invalid("缺少有效的生成请求标识");
        }
        if (!validSourceId(context.agentRunId(), 64) || !validSourceId(context.toolCallId(), 128)) {
            throw GenerationJobException.invalid(
                    "智能体运行标识最多 64 字符，原工具调用标识最多 128 字符，均须为 ASCII 字符");
        }
        if (users.lockEnabled(context.actor().userId()) == null) {
            throw new GenerationJobException(403, "GENERATION_ACCESS_DENIED", "当前用户不可用");
        }
    }

    private boolean validSourceId(String value, int limit) {
        return value == null || value.length() <= limit && value.chars().allMatch(character -> character <= 127);
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public void requireNotWithdrawn(String requestId, ResourceCommandContext context) {
        if (withdrawals.selectKind(scope(context), requestId) != null) {
            throw new GenerationJobException(410, "GENERATION_REQUEST_WITHDRAWN", "此提交已撤回，不会创建新的生成任务");
        }
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public void withdraw(String requestId, GenerationKind kind, ResourceCommandContext context) {
        String scope = scope(context);
        String previousKind = withdrawals.selectKind(scope, requestId);
        if (previousKind != null) {
            if (!previousKind.equals(kind.name())) {
                throw GenerationJobException.invalid("请求标识属于其他生成类别");
            }
            return;
        }
        withdrawals.insert(scope, requestId, kind.name(), context.actor().userId(), context.actor().teamId());
    }
}
