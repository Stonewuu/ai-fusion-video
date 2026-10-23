package com.stonewu.fusion.service.canvas.agent.node;

import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.mapper.canvas.CanvasGenerationUsageMapper;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.VideoModelRules;
import com.stonewu.fusion.service.generation.job.GenerationModelSnapshot;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Objects;

@Component
@RequiredArgsConstructor
public class CanvasAgentNodeGeneration {
    private final CanvasAgentNodePolicy policy;
    private final CanvasRunRepository runs;
    private final GenerationModelSnapshot models;
    private final CanvasJson json;
    private final CanvasGenerationUsageMapper usages;

    public boolean scoped(ToolExecutionContext context) {
        return context.getCanvasScope() != null && context.getCanvasScope().nodeRunId() != null;
    }

    public AiModel selected(ToolExecutionContext context, GenerationKind kind) {
        var job = policy.job(context.requireMutationContext(), false);
        if (job == null) {
            throw new BusinessException(403, "专业智能体的生成范围不存在");
        }
        return selected(job, kind);
    }

    public AiModel require(ToolExecutionContext context, GenerationKind kind) {
        var model = selected(context, kind);
        if (model == null) {
            throw new BusinessException(403, "本专业节点没有授权此生成能力；请修改节点模型和预算后重新运行");
        }
        return model;
    }

    private AiModel selected(CanvasAgentNodeRepository.Job job, GenerationKind kind) {
        Long selectedId = kind == GenerationKind.IMAGE ? job.scope().imageModelId() : job.scope().videoModelId();
        if (selectedId == null) {
            return null;
        }
        var node = runs.node(job.nodeRunId(), false);
        String field = kind == GenerationKind.IMAGE ? "imageModelId" : "videoModelId";
        var frozen = json.read(node.plan().environment().path(field).toString(), AiModel.class);
        if (!Objects.equals(selectedId, frozen.getId())) {
            throw new BusinessException(409, "专业智能体的生成模型快照不一致");
        }
        return models.frozen(frozen, kind, node.plan().environment().path(field + "ApiHash").asText());
    }

    /**
     * 受理事务持有专业节点范围行锁，所有并行子 Agent 共用同一份数量预算。
     */
    public String admit(ResourceCommandContext context, GenerationKind kind, AiModel model, int count, Integer seconds,
                        String nodeRunId) {
        var job = policy.job(context, true);
        if (job == null) {
            return nodeRunId;
        }
        var allowed = selected(job, kind);
        if (allowed == null || !json.hash(allowed).equals(json.hash(model))) {
            throw new BusinessException(403, "生成模型不在专业节点确认的范围内");
        }
        if (nodeRunId != null && !nodeRunId.equals(job.nodeRunId())) {
            throw new BusinessException(403, "不能替其他画布节点提交生成");
        }
        if (kind == GenerationKind.VIDEO && (seconds == null || seconds <= 0)) {
            throw new BusinessException(400, "专业节点生成视频必须明确时长");
        }
        long used = usages.selectNodeUsage(job.nodeRunId(), kind.name()).stream().mapToLong(usage -> {
            var input = json.read(usage.inputJson(), ObjectNode.class);
            var frozen = json.read(usage.modelSnapshotJson(), AiModel.class);
            int duration = kind == GenerationKind.VIDEO
                    ? VideoModelRules.budgetSeconds(JSONUtil.parseObj(frozen.getConfig()),
                    input.path("duration").asInt()) : 1;
            return Math.multiplyExact(input.path("count").asLong(), duration);
        }).sum();
        long requested = (long) count * (kind == GenerationKind.VIDEO ? seconds : 1);
        int limit = kind == GenerationKind.IMAGE ? job.scope().maxImages() : job.scope().maxVideoSeconds();
        if (requested < 1 || requested + used > limit) {
            throw new BusinessException(409, "专业智能体生成用量超过本次确认的数量上限，未提交新请求");
        }
        return job.nodeRunId();
    }
}
