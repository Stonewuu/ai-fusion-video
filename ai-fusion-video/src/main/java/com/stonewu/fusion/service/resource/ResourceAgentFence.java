package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * 业务事务持有 Agent Run 行锁，过期工作者无法在新 owner 接管后继续提交。
 */
@Component
@RequiredArgsConstructor
public class ResourceAgentFence {
    private final AgentRunMapper runs;

    public void requireActive(ResourceCommandContext context, Long projectId) {
        if (!"agent".equals(context.origin())) {
            return;
        }
        if (runs.lockActiveMutation(context, projectId) == null) {
            throw new CanvasException(409, CanvasErrorCode.ACCESS_DENIED,
                    "智能体运行已停止、租约失效或项目范围不符", Map.of("agentRunId", context.agentRunId()));
        }
    }
}
