package com.stonewu.fusion.service.canvas.agent;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeTypes;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Objects;
import java.util.Set;

/**
 * 子任务的事件、审批与正文使用自身日志；父任务只在结果确定后恢复，不占用父 owner。
 */
@Component
@RequiredArgsConstructor
public class CanvasChildAwait {
    private final AgentRunMapper runs;
    private final AgentMessageMapper messages;
    private final AgentRunRepository agents;
    private final CanvasQueryService canvases;
    private final CanvasJson json;

    public Object poll(String parentRunId, String toolCallId, String toolName, CanvasAgentScope scope,
                       CanvasActor actor) {
        if (!CanvasAgentNodeTypes.CHILD_TOOLS.contains(toolName) || scope == null || scope.nodeRunId() == null) {
            throw new BusinessException(403, "此子任务没有专业画布运行范围");
        }
        canvases.snapshot(scope.canvasId(), actor);
        var ids = runs.selectList(new LambdaQueryWrapper<AgentRun>().select(AgentRun::getRunId)
                .eq(AgentRun::getParentRunId, parentRunId).eq(AgentRun::getParentToolCallId, toolCallId));
        if (ids.size() != 1) {
            throw new BusinessException(409, "子任务尚未可靠受理，不能报告其已完成");
        }
        var child = agents.findRun(ids.getFirst().getRunId());
        var inherited = json.read(child.getCanvasScopeJson(), CanvasAgentScope.class);
        if (!Objects.equals(scope, inherited) || !Objects.equals(child.getUserId(), actor.userId())
                || !Objects.equals(child.getActorTeamId(), actor.teamId()) || !toolName.equals(child.getAgentName())) {
            throw new BusinessException(403, "子任务不属于当前画布、工具调用或身份");
        }
        if (!Set.of("COMPLETED", "FAILED", "CANCELLED")
                .contains(child.getStatus()) || child.getProjectionCompletedAt() == null) {
            return null;
        }
        var content = messages.selectList(new LambdaQueryWrapper<AgentMessage>().select(AgentMessage::getContent)
                        .eq(AgentMessage::getRunId, child.getRunId()).eq(AgentMessage::getRole, "assistant")
                        .isNotNull(AgentMessage::getContent).orderByAsc(AgentMessage::getMessageOrder))
                .stream().map(AgentMessage::getContent).toList();
        var result = new LinkedHashMap<String, Object>();
        result.put("status", "COMPLETED".equals(child.getStatus()) ? "success" : "error");
        result.put("childStatus", child.getStatus());
        result.put("childRunId", child.getRunId());
        result.put("parentRunId", parentRunId);
        result.put("parentToolCallId", toolCallId);
        result.put("agentName", toolName);
        result.put("executionMode", "PLATFORM_MANAGED_CHILD_RUN");
        result.put("resultDelivery", "PARENT_TOOL_RESULT");
        result.put("result", content.isEmpty() ? null : content.getLast());
        result.put("error", child.getErrorMessage());
        return result;
    }
}
