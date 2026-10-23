package com.stonewu.fusion.service.ai.tool.generation;

import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.GenerationItemViews;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.job.GenerationJobException;
import com.stonewu.fusion.service.generation.job.GenerationJobQueries;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Objects;

@Component
@RequiredArgsConstructor
public class GetGenerationTaskToolExecutor implements ToolExecutor {
    private final GenerationJobQueries queries;
    private final GenerationJobRepository jobs;
    private final ImageGenerationService images;
    private final VideoGenerationService videos;
    private final CanvasJson json;
    private final AgentRunMapper agents;

    @Override
    public String getToolName() {
        return "get_generation_task";
    }

    @Override
    public String getDisplayName() {
        return "查询生成任务";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public boolean isConcurrencySafe() {
        return true;
    }

    @Override
    public String getToolDescription() {
        return "读取 generate_image 或 generate_video 返回的 submissionId，获取真实状态及已托管的全部结果。等待中不产生新的生成请求，不要反复查询或伪造完成。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {"type":"object","properties":{"submissionId":{"type":"string","description":"提交工具返回的受理标识"}},"required":["submissionId"],"additionalProperties":false}
                """;
    }

    @Override
    public boolean isEnabled() {
        return true;
    }

    @Override
    public String execute(String input, ToolExecutionContext context) {
        var actor = context.requireMutationContext();
        String id = JSONUtil.parseObj(input).getStr("submissionId");
        var view = queries.get(id, actor.actor());
        var submission = jobs.submission(id);
        if (actor.agentRunId() != null) {
            var agent = agents.selectOne(new LambdaQueryWrapper<AgentRun>()
                    .select(AgentRun::getRunId, AgentRun::getProjectId)
                    .eq(AgentRun::getRunId, actor.agentRunId()).eq(AgentRun::getUserId, actor.actor().userId()));
            if (agent == null || (agent.getProjectId() != null && !Objects.equals(agent.getProjectId(),
                    submission.projectId()))) {
                throw new GenerationJobException(403, "GENERATION_SCOPE_MISMATCH",
                        "不能跨越当前智能体的项目范围读取任务");
            }
        }
        Object outputs = submission.kind() == GenerationKind.IMAGE
                ? images.listItems(submission.taskId()).stream()
                .map(item -> GenerationItemViews.project(item, true, images.getById(submission.taskId()).getStatus()))
                .toList()
                : videos.listItems(submission.taskId()).stream()
                .map(item -> GenerationItemViews.project(item, true, videos.getById(submission.taskId()).getStatus()))
                .toList();
        return json.write(Map.of("state", view.state(), "execution", view, "outputs", outputs));
    }
}
