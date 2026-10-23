package com.stonewu.fusion.service.canvas.agent.node;

import java.util.List;

import com.stonewu.fusion.mapper.canvas.CanvasAgentNodeJobMapper;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunExecutors.Executor;
import com.stonewu.fusion.service.canvas.run.CanvasRunExecutors.Outcome;
import com.stonewu.fusion.service.generation.job.GenerationJobCommands;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.ObjectProvider;

import org.springframework.stereotype.Component;

import java.util.Set;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

@Component
@RequiredArgsConstructor
public class CanvasAgentNodeExecutor implements Executor {
    private final CanvasAgentNodeJobs submissions;
    private final CanvasAgentNodeRepository jobs;
    private final AgentRunRepository agents;
    private final CanvasAgentNodeOutputs outputs;
    private final GenerationJobRepository generations;
    private final ObjectProvider<GenerationJobCommands> generationCommands;
    private final CanvasAgentNodeJobMapper nodes;
    private final GenerationSubmissionMapper submissionsMapper;

    @Override
    public Set<CanvasNodeKind> kinds() {
        return CanvasAgentNodeTypes.KINDS;
    }

    @Override
    public Outcome step(Run run, NodeRun node, Claim claim) {
        if (node.externalId() == null) {
            return new Outcome(State.RUNNING, Ports.empty(), "AGENT", submissions.submit(run, node), null);
        }
        var job = jobs.find(node.id(), false);
        if (job == null || !node.externalId().equals(job.agentRunId())) {
            throw CanvasRunException.invalid("智能体与节点执行关联不一致");
        }
        var agent = agents.findRun(job.agentRunId());
        State state = switch (agent.getStatus()) {
            case "COMPLETED" -> State.SUCCEEDED;
            case "FAILED" -> State.FAILED;
            case "CANCELLED" -> State.CANCELLED;
            case "WAITING_CONFIRMATION" -> State.WAITING_CONFIRMATION;
            case "CANCEL_REQUESTED" -> State.CANCEL_REQUESTED;
            default -> State.RUNNING;
        };
        var children = nodes.childStates(node.id(), agent.getRunId());
        if (children.contains("WAITING_CONFIRMATION") && !state.terminal()) {
            state = State.WAITING_CONFIRMATION;
        }
        if (state == State.SUCCEEDED && children.stream()
                .anyMatch(value -> Set.of("FAILED", "CANCELLED").contains(value))) {
            state = State.PARTIAL_FAILED;
        }
        if (state.terminal() && children.stream()
                .anyMatch(value -> !Set.of("COMPLETED", "FAILED", "CANCELLED").contains(value))) {
            if (state != State.SUCCEEDED && state != State.PARTIAL_FAILED) {
                jobs.requestCancel(node.id());
            }
            state = children.contains("WAITING_CONFIRMATION") ? State.WAITING_CONFIRMATION : State.RUNNING;
        }
        String error = agent.getErrorMessage();
        for (String id : generationIds(node.id())) {
            var generation = generations.submission(id);
            if (generation.state().equals("UNKNOWN_SUBMISSION")) {
                state = State.UNKNOWN_SUBMISSION;
                error = "生成服务未返回完整回执，请在本节点的生成任务中核实原请求。";
            } else if (generation.state().equals("NEEDS_REVIEW") && state != State.UNKNOWN_SUBMISSION) {
                state = State.NEEDS_REVIEW;
                error = "生成结果需要处理，请打开本节点的生成任务查看详情。";
            } else if (!Set.of("SUCCEEDED", "PARTIAL_FAILED", "FAILED", "CANCELLED")
                    .contains(generation.state()) && state.terminal()) {
                state = State.RUNNING;
            } else if (generation.state().equals("PARTIAL_FAILED") && state == State.SUCCEEDED) {
                state = State.PARTIAL_FAILED;
            } else if (Set.of("FAILED", "CANCELLED").contains(generation.state()) && state == State.SUCCEEDED) {
                state = State.PARTIAL_FAILED;
            }
        }
        if ((state == State.SUCCEEDED || state == State.PARTIAL_FAILED) && agent.getProjectionCompletedAt() == null) {
            state = State.RUNNING;
        }
        long uncertain = nodes.unresolvedModelIds(node.id()).size();
        if (uncertain > 0) {
            return new Outcome(State.UNKNOWN_SUBMISSION, outputs.preview(run, node), "AGENT", job.agentRunId(),
                    "模型连接已中断，暂时无法确认本次调用是否结束。请打开本节点的任务详情处理。");
        }
        try {
            var result = state == State.SUCCEEDED || state == State.PARTIAL_FAILED ? outputs.collect(run, node,
                    job) : outputs.preview(run, node);
            return new Outcome(state, result, "AGENT", job.agentRunId(), error);
        } catch (BusinessException invalidOutput) {
            if (invalidOutput.getCode() != 400 && invalidOutput.getCode() != 409) {
                throw invalidOutput;
            }
            return new Outcome(State.FAILED, Ports.empty(), "AGENT", job.agentRunId(), invalidOutput.getMessage());
        }
    }

    @Override
    public void cancel(Run run, NodeRun node) {
        jobs.requestCancel(node.id());
        for (String id : generationIds(node.id())) {
            var submission = generations.submission(id);
            generationCommands.getObject().cancel(submission.kind(), submission.taskId(), run.actor());
        }
    }

    private List<String> generationIds(String nodeRunId) {
        return submissionsMapper.selectList(Wrappers.<GenerationSubmissionEntity>lambdaQuery()
                        .select(GenerationSubmissionEntity::getId)
                        .eq(GenerationSubmissionEntity::getCanvasNodeRunId, nodeRunId))
                .stream().map(GenerationSubmissionEntity::getId).toList();
    }
}
