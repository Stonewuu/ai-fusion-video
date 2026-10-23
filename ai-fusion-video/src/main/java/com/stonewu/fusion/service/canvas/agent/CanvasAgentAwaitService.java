package com.stonewu.fusion.service.canvas.agent;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunService;
import io.agentscope.core.tool.ToolSuspendException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.Set;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.State;
import static com.stonewu.fusion.service.canvas.run.CanvasRunData.View;

@Service
@RequiredArgsConstructor
public class CanvasAgentAwaitService {
    private final CanvasAgentAccess access;
    private final CanvasAgentRunTools tools;
    private final CanvasRunService runs;
    private static final Set<State> ATTENTION = Set.of(State.WAITING_SELECTION, State.WAITING_CONFIRMATION,
            State.UNKNOWN_SUBMISSION, State.NEEDS_REVIEW);

    public Object await(String id, ToolExecutionContext context) {
        var session = access.require(context);
        tools.requireBound(id, session);
        var result = poll(id, session.scope(), session.context().actor());
        if (result != null) {
            return result;
        }
        // SDK 持久化未完成工具调用并结束本次流；Supervisor 提交 WAITING_EXTERNAL 后释放 owner。
        throw new ToolSuspendException("画布任务仍在运行；已交由持久化等待，完成或需要处理时继续。");
    }

    public Object poll(String id, CanvasAgentScope scope, CanvasActor actor) {
        var view = runs.view(id, actor);
        if (!view.canvasId().equals(scope.canvasId())) {
            throw new BusinessException(403, "运行不属于当前画布上下文");
        }
        return ready(view) ? CanvasAgentRunTools.summary(view, 0, 20) : null;
    }

    public static boolean ready(View view) {
        return view.status().terminal() || ATTENTION.contains(view.status())
                || view.nodes().stream().anyMatch(node -> ATTENTION.contains(node.status()));
    }
}
