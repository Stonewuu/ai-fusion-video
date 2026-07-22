package com.stonewu.fusion.service.ai.agentscope;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.config.AgentScopeRuntimeProperties;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.controller.ai.vo.AiChatReqVO;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.AiAgentService;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextRequest;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.message.AgentScopeMessageMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.AgentExecutionRuntimeContextRequests;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.AgentRunQueryService;
import com.stonewu.fusion.service.ai.run.AgentRunReplayService;
import com.stonewu.fusion.service.ai.run.AgentRuntimeInstanceIdentity;
import com.stonewu.fusion.service.ai.run.AgentRuntimeMetrics;
import com.stonewu.fusion.service.ai.run.RunExecutionSupervisor;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.StartAgentExecutionCommand;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import io.agentscope.core.message.UserMessage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentScopePipelineRunServiceTests {

    private final AgentRuntimeSchedulers schedulers = schedulers();

    @AfterEach
    void closeSchedulers() {
        schedulers.close();
    }

    @Test
    void startsOnlyTheDurableHarnessExecutionWithStrongMessages() {
        AiModelService models = mock(AiModelService.class);
        AiAgentService agents = mock(AiAgentService.class);
        AgentConversationService conversations = mock(AgentConversationService.class);
        AgentKernelSpecFactory specs = mock(AgentKernelSpecFactory.class);
        AgentKernelSnapshotBuilder snapshots = mock(AgentKernelSnapshotBuilder.class);
        AgentRunCoordinator coordinator = mock(AgentRunCoordinator.class);
        AgentExecutionRuntimeContextRequests runtimeContexts =
                mock(AgentExecutionRuntimeContextRequests.class);
        RunExecutionSupervisor supervisor = mock(RunExecutionSupervisor.class);
        AgentRunQueryService queries = mock(AgentRunQueryService.class);
        AgentRunReplayService replay = mock(AgentRunReplayService.class);
        AgentRuntimeInstanceIdentity identity = mock(AgentRuntimeInstanceIdentity.class);
        AgentScopeV2Properties properties = new AgentScopeV2Properties();
        AgentKernelSpec spec = mock(AgentKernelSpec.class);
        AgentScopeRuntimeContextRequest runtime = mock(AgentScopeRuntimeContextRequest.class);
        AiModel model = AiModel.builder()
                .id(7L)
                .code("model")
                .status(1)
                .build();
        AgentKernelSnapshot snapshot = snapshot();

        when(models.getDefaultByType(1)).thenReturn(model);
        when(specs.createRoot(any(AiChatReqVO.class), eq(model), any(String.class)))
                .thenReturn(spec);
        when(spec.agentDefinitionStableKey()).thenReturn("ai_assistant_agent");
        when(snapshots.build(spec)).thenReturn(snapshot);
        when(identity.value()).thenReturn("node-1");
        when(coordinator.start(any(StartAgentRunCommand.class)))
                .thenAnswer(invocation -> {
                    StartAgentRunCommand command = invocation.getArgument(0);
                    return Mono.just(new StartedAgentRun(
                            command.runId(),
                            command.conversationId(),
                            command.agentStateSessionId(),
                            command.ownerInstanceId(),
                            1L,
                            command.deadline().minusSeconds(1),
                            command.deadline(),
                            snapshot,
                            1L));
                });
        when(runtimeContexts.forRoot(any(), eq("ai_assistant_agent"), isNull()))
                .thenReturn(Mono.just(runtime));
        when(supervisor.start(any(StartAgentExecutionCommand.class))).thenReturn(Mono.empty());

        AgentScopePipelineRunService service = new AgentScopePipelineRunService(
                models,
                agents,
                conversations,
                specs,
                snapshots,
                new AgentScopeMessageMapper(),
                coordinator,
                runtimeContexts,
                supervisor,
                queries,
                replay,
                identity,
                properties,
                schedulers,
                new ObjectMapper());
        AiChatReqVO request = new AiChatReqVO()
                .setConversationId("conversation-1")
                .setMessage("hello harness");

        StepVerifier.create(service.start(request, 42L))
                .assertNext(started -> assertThat(started.runId()).isNotBlank())
                .verifyComplete();

        ArgumentCaptor<StartAgentRunCommand> admission =
                ArgumentCaptor.forClass(StartAgentRunCommand.class);
        verify(coordinator).start(admission.capture());
        assertThat(admission.getValue().agentType()).isEqualTo("ai_assistant_agent");
        assertThat(admission.getValue().agentStateSessionId())
                .isEqualTo("afv:v2:conversation-1:ai_assistant_agent");
        assertThat(admission.getValue().userContent()).isEqualTo("hello harness");

        ArgumentCaptor<StartAgentExecutionCommand> execution =
                ArgumentCaptor.forClass(StartAgentExecutionCommand.class);
        verify(supervisor).start(execution.capture());
        assertThat(execution.getValue().messages()).singleElement()
                .isInstanceOfSatisfying(UserMessage.class, message ->
                        assertThat(message.getTextContent()).isEqualTo("hello harness"));
        assertThat(execution.getValue().kernelSpec()).isSameAs(spec);
        assertThat(execution.getValue().runtimeContextRequest()).isSameAs(runtime);
    }

    private AgentKernelSnapshot snapshot() {
        return new CanonicalAgentKernelSnapshotBuilder().build(
                new AgentKernelSnapshotPayload(
                        AgentKernelSnapshotPayload.CURRENT_SCHEMA_VERSION,
                        "ai_assistant_agent",
                        "assistant",
                        "test",
                        "system",
                        5,
                        "7",
                        1,
                        "openai",
                        "model",
                        JsonNodeFactory.instance.objectNode(),
                        List.of(),
                        "test"));
    }

    private AgentRuntimeSchedulers schedulers() {
        AgentScopeRuntimeProperties properties = new AgentScopeRuntimeProperties();
        properties.setStateThreads(1);
        properties.setJournalThreads(1);
        properties.setModelThreads(1);
        properties.setToolThreads(1);
        return new AgentRuntimeSchedulers(properties, AgentRuntimeMetrics.noop());
    }
}
