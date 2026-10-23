package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.canvas.run.CanvasRunData;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import com.stonewu.fusion.service.generation.job.GenerationSubmission;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.beans.factory.ObjectProvider;
import com.stonewu.fusion.mapper.canvas.CanvasAgentNodeJobMapper;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CanvasAgentNodeStateTests {
    @ParameterizedTest
    @ValueSource(strings = {"UNKNOWN_SUBMISSION", "NEEDS_REVIEW"})
    void unfinishedGenerationExplainsWhereToRecoverInsteadOfShowingAnUnexplainedAgentWait(String state) {
        var run = mock(CanvasRunData.Run.class);
        var node = mock(CanvasRunData.NodeRun.class);
        when(node.id()).thenReturn("node");
        when(node.externalId()).thenReturn("agent");
        var jobs = mock(CanvasAgentNodeRepository.class);
        var job = mock(CanvasAgentNodeRepository.Job.class);
        when(job.agentRunId()).thenReturn("agent");
        when(jobs.find("node", false)).thenReturn(job);
        var agents = mock(AgentRunRepository.class);
        var agent = mock(AgentRun.class);
        when(agent.getRunId()).thenReturn("agent");
        when(agent.getStatus()).thenReturn("WAITING_EXTERNAL");
        when(agents.findRun("agent")).thenReturn(agent);
        var nodes = mock(CanvasAgentNodeJobMapper.class);
        var submissions = mock(GenerationSubmissionMapper.class);
        when(submissions.selectList(any())).thenReturn(
                List.of(GenerationSubmissionEntity.builder().id("generation").build()));
        var generations = mock(GenerationJobRepository.class);
        var generation = mock(GenerationSubmission.class);
        when(generation.state()).thenReturn(state);
        when(generations.submission("generation")).thenReturn(generation);
        var outputs = mock(CanvasAgentNodeOutputs.class);
        when(outputs.preview(run, node)).thenReturn(CanvasRunData.Ports.empty());
        var executor = new CanvasAgentNodeExecutor(mock(CanvasAgentNodeJobs.class), jobs, agents, outputs,
                generations, mock(ObjectProvider.class), nodes, submissions);

        var result = executor.step(run, node, mock(CanvasRunData.Claim.class));

        assertThat(result.state().name()).isEqualTo(state);
        assertThat(result.error()).contains("生成任务").doesNotContain("设置", "审批");
        assertThat(result.outputs()).isEqualTo(CanvasRunData.Ports.empty());
    }
}
