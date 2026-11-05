package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.tool.generation.GetGenerationTaskToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
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
                "等待已受理的图片或视频任务完成，适用于资产、分镜和画布生成。等待释放执行线程并自动恢复；使用成功输出继续写回。"
                        + "SUBMITTING、排队、生成中都应继续等待同一 submissionId，不轮询也不重新提交。状态未知或需要核实时报告待核实，不能当作失败重试。",
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
        Object result = poll(canvases, jobs, query, json, input, context);
        if (result != null) {
            return result;
        }
        throw new ToolSuspendException("生成任务尚未结束，已交由持久化等待观察原请求。");
    }

    @Override
    public Object poll(ObjectNode input, ToolExecutionContext context) {
        return poll(canvases, jobs, query, json, input, context);
    }

    private static Object poll(CanvasQueryService canvases, GenerationJobRepository jobs,
                               GetGenerationTaskToolExecutor query,
                               CanvasJson json, ObjectNode input, ToolExecutionContext context) {
        if (context.requireMutationContext().agentRunId() == null) {
            throw new BusinessException(403, "等待生成结果需要智能体运行上下文");
        }
        // 查询统一校验用户、团队与当前项目范围，恢复后仍使用原任务身份。
        ObjectNode result = json.read(query.execute(json.write(input), context), ObjectNode.class);
        var scope = context.getCanvasScope();
        if (scope != null) {
            canvases.snapshot(scope.canvasId(), context.requireMutationContext().actor());
            if (scope.nodeRunId() != null && !scope.nodeRunId().equals(jobs.submission(text(input, "submissionId")).canvasNodeRunId())) {
                throw new BusinessException(403, "生成任务不属于本专业节点");
            }
        }
        if (!Set.of("SUCCEEDED", "PARTIAL_FAILED", "FAILED", "CANCELLED", "NEEDS_REVIEW", "UNKNOWN_SUBMISSION")
                .contains(result.path("state").asText())) {
            return null;
        }
        return result;
    }
}
