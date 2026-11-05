package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutorRegistry;
import com.stonewu.fusion.service.ai.run.AgentExecutionRuntimeContextRequests;
import com.stonewu.fusion.service.ai.run.AgentRuntimeInstanceIdentity;
import com.stonewu.fusion.service.ai.run.AgentWaitingStatePort;
import com.stonewu.fusion.service.ai.run.RunExecutionSupervisor;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.ToolManifestSnapshot;
import com.stonewu.fusion.service.ai.run.model.PendingExternalExecution;
import com.stonewu.fusion.service.ai.run.model.ResumeExternalCommand;
import com.stonewu.fusion.service.canvas.CanvasJson;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

class GenerationWaitBridgeTests {
    @Test
    @DisplayName("无画布任务释放 owner 后仍保留原运行身份读取，完成后只恢复原等待调用")
    void resumesOrdinaryGenerationUsingItsPersistedRunScopeWithoutAWriteLease() throws Exception {
        var mapper = new ObjectMapper();
        var json = new CanvasJson(mapper);
        var queue = mock(CanvasAgentWaitRepository.class);
        var tools = mock(ToolExecutorRegistry.class);
        var models = mock(AiModelService.class);
        var waiting = mock(AgentWaitingStatePort.class);
        var identity = mock(AgentRuntimeInstanceIdentity.class);
        when(identity.value()).thenReturn("waiter");
        var tool = mock(CanvasExternalTool.class);
        when(tools.findExecutor("await_generation_task")).thenReturn(tool);
        var model = AiModel.builder().id(7L).build();
        when(models.getById(7L)).thenReturn(model);
        var snapshot = new AgentKernelSnapshotPayload(2, "asset", "asset", "资产生成", "等待原任务", 20,
                "7", 1, "test", "test", mapper.createObjectNode(),
                List.of(new ToolManifestSnapshot("await_generation_task", "a".repeat(64), true, true, "1")), "1.3.1");
        var run = AgentRun.builder().runId("asset-run").userId(7L).actorTeamId(81L).projectId(5L)
                .status("WAITING_EXTERNAL").ownerInstanceId(null).ownerEpoch(1L)
                .deadlineAt(LocalDateTime.now(ZoneOffset.UTC).plusMinutes(2))
                .waitingToolCallId("reply").agentStateSessionId("session")
                .agentDefinitionSnapshotJson(json.write(snapshot)).build();
        when(queue.deferCheck("asset-run")).thenReturn(run);
        when(waiting.getPendingExternalAuthorized("asset-run", 7L, "reply")).thenReturn(Mono.just(
                new PendingExternalExecution("reply", "await_canvas_run",
                        "{\"toolCalls\":[{\"id\":\"wait-call\",\"name\":\"await_generation_task\",\"input\":{\"submissionId\":\"accepted\"}}]}",
                        Instant.now().plusSeconds(120))));
        when(tool.poll(any(), any())).thenReturn(null, Map.of("state", "SUCCEEDED", "outputs", List.of()));
        when(waiting.resumeExternal(any())).thenReturn(Mono.empty());
        var bridge = new CanvasAgentWaitBridge(mock(CanvasChildAwait.class), queue, mock(CanvasAgentScopeService.class),
                tools, models, waiting, mock(AgentExecutionRuntimeContextRequests.class), mock(RunExecutionSupervisor.class),
                mock(AgentKernelSnapshotBuilder.class), identity, new AgentScopeV2Properties(), json, mapper);
        try {
            bridge.check("asset-run");
            verify(waiting, never()).resumeExternal(any());
            bridge.check("asset-run");
            var contexts = ArgumentCaptor.forClass(ToolExecutionContext.class);
            verify(tool, times(2)).poll(any(), contexts.capture());
            var context = contexts.getValue();
            assertThat(context.getCanvasScope()).isNull();
            assertThat(context.getAgentSessionId()).isEqualTo("session");
            assertThat(context.requireMutationContext().agentRunId()).isEqualTo("asset-run");
            assertThat(context.requireMutationContext().actor().teamId()).isEqualTo(81L);
            assertThat(context.requireMutationContext().ownerInstanceId()).isNull();
            var resume = ArgumentCaptor.forClass(ResumeExternalCommand.class);
            verify(waiting).resumeExternal(resume.capture());
            assertThat(resume.getValue().runId()).isEqualTo("asset-run");
            assertThat(resume.getValue().toolCallId()).isEqualTo("reply");
            var delivered = mapper.readTree(resume.getValue().resultPayloadJson()).path("results").get(0);
            assertThat(delivered.path("toolCallId").asText()).isEqualTo("wait-call");
            assertThat(delivered.path("result").path("state").asText()).isEqualTo("SUCCEEDED");
        } finally {
            bridge.close();
        }
    }
}
