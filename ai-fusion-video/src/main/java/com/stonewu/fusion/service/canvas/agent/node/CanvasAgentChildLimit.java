package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.service.ai.run.AgentChildAdmissionPolicy;
import com.stonewu.fusion.service.ai.run.model.StartChildAgentRunCommand;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScope;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.canvas.CanvasAgentNodeJobMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class CanvasAgentChildLimit implements AgentChildAdmissionPolicy {
    private final CanvasAgentNodeJobMapper jobs;
    private final AgentRunMapper runs;
    private final CanvasJson json;

    @Override
    public void admit(AgentRun parent, StartChildAgentRunCommand command) {
        if (parent.getCanvasScopeJson() == null) {
            return;
        }
        var scope = json.read(parent.getCanvasScopeJson(), CanvasAgentScope.class);
        if (scope.nodeRunId() == null) {
            return;
        }
        var limits = jobs.lockActiveScope(scope.nodeRunId());
        if (limits == null) {
            throw new BusinessException(409, "专业节点已停止，不能再启动子任务");
        }
        // 已受理的同一工具调用重放不重复消耗任务名额。
        long old = runs.selectCount(new LambdaQueryWrapper<AgentRun>().eq(AgentRun::getParentRunId, parent.getRunId())
                .eq(AgentRun::getParentToolCallId, command.parentToolCallId()));
        if (old == 1) {
            return;
        }
        var limit = json.read(limits, CanvasAgentNodeScope.class).maxAgentRuns();
        long used = jobs.countAgentRuns(scope.nodeRunId());
        if (used >= limit) {
            throw new BusinessException(409, "已达到本专业节点确认的智能体任务上限，未启动新的子任务");
        }
    }
}
