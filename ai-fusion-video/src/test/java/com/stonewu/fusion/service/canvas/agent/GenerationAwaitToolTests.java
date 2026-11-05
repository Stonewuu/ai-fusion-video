package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.tool.generation.GetGenerationTaskToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import com.stonewu.fusion.service.generation.job.GenerationSubmission;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import io.agentscope.core.tool.ToolSuspendException;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.*;

class GenerationAwaitToolTests {
    private final CanvasJson json = new CanvasJson(new ObjectMapper());
    private final CanvasQueryService canvases = mock(CanvasQueryService.class);
    private final GenerationJobRepository jobs = mock(GenerationJobRepository.class);
    private final GetGenerationTaskToolExecutor query = mock(GetGenerationTaskToolExecutor.class);
    private final CanvasGenerationAwaitTool tool = new CanvasGenerationAwaitTool(canvases, jobs, query, json);
    private final String input = "{\"submissionId\":\"accepted\"}";
    private final ToolExecutionContext context = ToolExecutionContext.forInvocation(7L, 1, 7L,
            new ResourceCommandContext(new CanvasActor(7, null), "agent", null, "asset-run", "wait-call", "owner", 1L));

    @Test
    @DisplayName("普通资产任务在供应商提交中暂停等待，完成后返回原任务的输出")
    void ordinaryGenerationSuspendsAndResumesTheSameSubmission() {
        when(query.execute(input, context)).thenReturn("{\"state\":\"RUNNING\",\"execution\":{\"items\":[{\"phase\":\"SUBMITTING\"}]}}");
        assertThatThrownBy(() -> tool.execute(input, context)).isInstanceOf(ToolSuspendException.class);
        assertThat(tool.poll(json.read(input, ObjectNode.class), context)).isNull();
        when(query.execute(input, context)).thenReturn("{\"state\":\"SUCCEEDED\",\"outputs\":[{\"imageUrl\":\"media://image\"}]}");
        var result = json.read(tool.execute(input, context), ObjectNode.class);
        assertThat(result.path("state").asText()).isEqualTo("SUCCEEDED");
        assertThat(result.path("outputs").get(0).path("imageUrl").asText()).isEqualTo("media://image");
        verify(query, times(3)).execute(input, context);
        verifyNoInteractions(canvases, jobs);
    }

    @Test
    @DisplayName("等待结果继续采用查询工具的用户和项目权限，不能丢失原运行范围")
    void permissionFailureIsReportedInsteadOfBecomingAnEndlessWait() {
        when(query.execute(input, context)).thenThrow(new BusinessException(403, "不能跨越当前智能体的项目范围读取任务"));
        var result = json.read(tool.execute(input, context), ObjectNode.class);
        assertThat(result.path("status").asText()).isEqualTo("error");
        assertThat(result.path("code").intValue()).isEqualTo(403);
        assertThatThrownBy(() -> tool.poll(json.read(input, ObjectNode.class), context)).isInstanceOf(BusinessException.class);
        verifyNoInteractions(jobs);
    }

    @Test
    @DisplayName("画布专业节点仍不能等待其他节点提交的生成任务")
    void canvasNodeBoundaryRemainsEnforced() {
        var scope = mock(CanvasAgentScope.class);
        when(scope.canvasId()).thenReturn("canvas");
        when(scope.nodeRunId()).thenReturn("node");
        context.setCanvasScope(scope);
        var submission = mock(GenerationSubmission.class);
        when(submission.canvasNodeRunId()).thenReturn("other-node");
        when(jobs.submission("accepted")).thenReturn(submission);
        when(query.execute(input, context)).thenReturn("{\"state\":\"SUCCEEDED\"}");
        assertThat(json.read(tool.execute(input, context), ObjectNode.class).path("code").intValue()).isEqualTo(403);
    }

    @Test
    @DisplayName("状态未知或需要核实时返回原状态，不把请求当作失败重新生成")
    void uncertainResultsRemainExplicit() {
        for (String state : new String[]{"UNKNOWN_SUBMISSION", "NEEDS_REVIEW", "PARTIAL_FAILED", "FAILED", "CANCELLED"}) {
            when(query.execute(input, context)).thenReturn("{\"state\":\"" + state + "\"}");
            assertThat(json.read(tool.execute(input, context), ObjectNode.class).path("state").asText()).isEqualTo(state);
        }
    }
}
