package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.tool.generation.GetGenerationTaskToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import io.agentscope.core.tool.ToolSuspendException;

import java.util.Map;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.agent.CanvasAgentSchema.*;

public final class CanvasGenerationAwaitTool extends CanvasAgentTool implements CanvasExternalTool {
    private final CanvasQueryService canvases;
    private final GenerationJobRepository jobs;
    private final GetGenerationTaskToolExecutor query;
    private final CanvasJson json;

    public CanvasGenerationAwaitTool(CanvasQueryService canvases, GenerationJobRepository jobs,
                                     GetGenerationTaskToolExecutor query, CanvasJson json) {
        super("await_generation_task", "等待生成结果",
                "持久化等待本专业画布节点的生成任务结束或需要核实。等待释放执行线程；使用实际成功输出继续业务写回，不循环查询也不重新生成。",
                schema(Map.of("submissionId", string("生成工具返回的可靠受理标识")), "submissionId"), true, json,
                (input, context) -> await(canvases, jobs, query, json, input, context));
        this.canvases = canvases;
        this.jobs = jobs;
        this.query = query;
        this.json = json;
    }

    private static Object await(CanvasQueryService canvases, GenerationJobRepository jobs,
                                GetGenerationTaskToolExecutor query,
                                CanvasJson json, ObjectNode input, ToolExecutionContext context) {
        Object result = poll(canvases, jobs, query, json, input, context.getCanvasScope(),
                context.requireMutationContext().actor());
        if (result != null) {
            return result;
        }
        throw new ToolSuspendException("生成任务尚未结束，已交由持久化等待观察原请求。");
    }

    @Override
    public Object poll(ObjectNode input, CanvasAgentScope scope, CanvasActor actor, AiModel model) {
        return poll(canvases, jobs, query, json, input, scope, actor);
    }

    private static Object poll(CanvasQueryService canvases, GenerationJobRepository jobs,
                               GetGenerationTaskToolExecutor query,
                               CanvasJson json, ObjectNode input, CanvasAgentScope scope, CanvasActor actor) {
        if (scope == null || scope.nodeRunId() == null) {
            throw new BusinessException(403, "此等待工具仅用于可靠受理的专业画布节点");
        }
        canvases.snapshot(scope.canvasId(), actor);
        var job = jobs.submission(text(input, "submissionId"));
        if (!scope.nodeRunId().equals(job.canvasNodeRunId())) {
            throw new BusinessException(403, "生成任务不属于本专业节点");
        }
        if (!Set.of("SUCCEEDED", "PARTIAL_FAILED", "FAILED", "CANCELLED", "NEEDS_REVIEW", "UNKNOWN_SUBMISSION")
                .contains(job.state())) {
            return null;
        }
        var context = ToolExecutionContext.forInvocation(actor.userId(), actor.teamId() == null ? 1 : 2,
                actor.teamId() == null ? actor.userId() : actor.teamId(),
                ResourceCommandContext.user(actor, scope.canvasId()));
        return json.read(query.execute(json.write(input), context), ObjectNode.class);
    }
}
