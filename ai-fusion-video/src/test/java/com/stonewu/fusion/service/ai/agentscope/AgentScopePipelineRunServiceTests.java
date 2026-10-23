package com.stonewu.fusion.service.ai.agentscope;

import com.stonewu.fusion.security.SecurityUserDetails;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScopeService;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.config.AgentScopeRuntimeProperties;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.controller.ai.vo.AiChatReqVO;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.AgentMessageService;
import com.stonewu.fusion.service.ai.AiAgentService;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextRequest;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.message.AgentScopeMessageMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.run.AgentExecutionRuntimeContextRequests;
import com.stonewu.fusion.service.ai.run.AgentExecutionFactory;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.AgentRunQueryService;
import com.stonewu.fusion.service.ai.run.AgentRunReplayService;
import com.stonewu.fusion.service.ai.run.AgentRuntimeInstanceIdentity;
import com.stonewu.fusion.service.ai.run.AgentRuntimeMetrics;
import com.stonewu.fusion.service.ai.run.AgentStateSessionIds;
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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentScopePipelineRunServiceTests {

    private final AgentRuntimeSchedulers schedulers = schedulers();

    @AfterEach
    void closeSchedulers() {
        schedulers.close();
        SecurityContextHolder.clearContext();
    }

    @ParameterizedTest
    @ValueSource(strings = {"initial", "new_generation", "existing_generation"})
    void startsOnlyTheDurableHarnessExecutionWithStrongMessages(String generation) {
        AiModelService models = mock(AiModelService.class);
        AiAgentService agents = mock(AiAgentService.class);
        AgentConversationService conversations = mock(AgentConversationService.class);
        AgentMessageService persistedMessages = mock(AgentMessageService.class);
        AgentKernelSpecFactory specs = mock(AgentKernelSpecFactory.class);
        AgentKernelSnapshotBuilder snapshots = mock(AgentKernelSnapshotBuilder.class);
        AgentRunCoordinator coordinator = mock(AgentRunCoordinator.class);
        AgentExecutionRuntimeContextRequests runtimeContexts =
                mock(AgentExecutionRuntimeContextRequests.class);
        AgentExecutionFactory executionFactory = mock(AgentExecutionFactory.class);
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
                .supportReasoning(true)
                .reasoningEffortLevels(List.of("high", "low"))
                .build();
        AgentKernelSnapshot snapshot = snapshot();

        when(models.getDefaultByType(1)).thenReturn(model);
        when(specs.createRoot(any(AiChatReqVO.class), any(AiModel.class), any(String.class), eq(42L)))
                .thenReturn(spec);
        when(spec.agentDefinitionStableKey()).thenReturn("ai_assistant_agent");
        when(snapshots.build(spec)).thenReturn(snapshot);
        when(identity.value()).thenReturn("node-1");
        when(persistedMessages.listRootHistory("conversation-1")).thenReturn(List.of(
                AgentMessage.builder().role("user").content("已经要求创建灯塔节点").build(),
                AgentMessage.builder().role("tool").toolCallId("created-node").toolName("apply_canvas_operations")
                        .toolStatus("success").content("{\"nodeId\":\"actual-node\"}").build()));
        when(coordinator.start(any(StartAgentRunCommand.class)))
                .thenAnswer(invocation -> {
                    StartAgentRunCommand command = invocation.getArgument(0);
                    String stateSession = switch (generation) {
                        case "new_generation" -> AgentStateSessionIds.recoveryGeneration(
                                command.conversationId(), command.agentType(), command.runId());
                        case "existing_generation" -> AgentStateSessionIds.recoveryGeneration(
                                command.conversationId(), command.agentType(), "previous-completed-run");
                        default -> command.stateSessionCandidate();
                    };
                    return Mono.just(new StartedAgentRun(
                            command.runId(),
                            command.conversationId(),
                            stateSession,
                            command.ownerInstanceId(),
                            1L,
                            command.deadline().minusSeconds(1),
                            command.deadline(),
                            snapshot,
                            1L));
                });
        when(runtimeContexts.forRoot(
                any(), eq("ai_assistant_agent"), isNull(), eq(ToolExecutionMode.DEFAULT)))
                .thenReturn(Mono.just(runtime));
        when(supervisor.start(any(StartAgentExecutionCommand.class))).thenReturn(Mono.empty());

        AgentScopePipelineRunService service = new AgentScopePipelineRunService(
                models,
                agents,
                conversations,
                persistedMessages,
                specs,
                snapshots,
                new AgentScopeMessageMapper(),
                coordinator,
                runtimeContexts,
                executionFactory,
                supervisor,
                queries,
                replay,
                identity,
                properties,
                schedulers,
                new ObjectMapper(),
                mock(CanvasAgentScopeService.class));
        AiChatReqVO request = new AiChatReqVO()
                .setConversationId("conversation-1")
                .setMessage("hello harness")
                .setReasoningEffort("high")
                .setToolExecutionMode(ToolExecutionMode.DEFAULT.name())
                .setEnabledSkills(List.of("story-review"));

        SecurityUserDetails principal = new SecurityUserDetails(42L, "creator", "test-only", 1, 77L, List.of());
        SecurityContextHolder.getContext().setAuthentication(new UsernamePasswordAuthenticationToken(principal, null, principal.getAuthorities()));
        Mono<StartedAgentRun> starting = service.start(request, 42L);
        SecurityContextHolder.clearContext();
        StepVerifier.create(starting)
                .assertNext(started -> assertThat(started.runId()).isNotBlank())
                .verifyComplete();

        ArgumentCaptor<StartAgentRunCommand> admission =
                ArgumentCaptor.forClass(StartAgentRunCommand.class);
        verify(coordinator).start(admission.capture());
        assertThat(admission.getValue().agentType()).isEqualTo("ai_assistant_agent");
        assertThat(admission.getValue().actorTeamId()).isEqualTo(77L);
        assertThat(admission.getValue().stateSessionCandidate())
                .isEqualTo("afv:v2:conversation-1:ai_assistant_agent");
        assertThat(admission.getValue().userContent()).isEqualTo("hello harness");

        ArgumentCaptor<String> systemPrompt = ArgumentCaptor.forClass(String.class);
        ArgumentCaptor<AiModel> effectiveModel = ArgumentCaptor.forClass(AiModel.class);
        verify(specs).createRoot(
                any(AiChatReqVO.class), effectiveModel.capture(), systemPrompt.capture(), eq(42L));
        assertThat(effectiveModel.getValue().getConfig()).contains("\"reasoningEffort\":\"high\"");
        assertThat(model.getConfig()).isNull();
        assertThat(request.getEnabledSkills()).containsExactly("story-review");
        verify(specs).createRoot(eq(request), any(AiModel.class), any(String.class), eq(42L));

        ArgumentCaptor<StartAgentExecutionCommand> execution =
                ArgumentCaptor.forClass(StartAgentExecutionCommand.class);
        verify(supervisor).start(execution.capture());
        assertThat(execution.getValue().messages()).hasSize(generation.equals("new_generation") ? 3 : 1);
        assertThat(execution.getValue().messages().getLast()).isInstanceOfSatisfying(UserMessage.class,
                message -> assertThat(message.getTextContent()).isEqualTo("hello harness"));
        if (generation.equals("new_generation")) {
            assertThat(execution.getValue().messages().getFirst().getTextContent()).isEqualTo("已经要求创建灯塔节点");
            assertThat(execution.getValue().messages().get(1).getTextContent()).contains("actual-node", "状态：success");
        }
        assertThat(execution.getValue().kernelSpec()).isSameAs(spec);
        assertThat(execution.getValue().runtimeContextRequest()).isSameAs(runtime);
    }

    @ParameterizedTest
    @ValueSource(strings = {"FAILED", "CANCELLED"})
    void continuesRetryableRunFromPersistedKernelAfterProcessRestart(String status) {
        AiModelService models = mock(AiModelService.class);
        AiAgentService agents = mock(AiAgentService.class);
        AgentConversationService conversations = mock(AgentConversationService.class);
        AgentMessageService persistedMessages = mock(AgentMessageService.class);
        AgentKernelSpecFactory specs = mock(AgentKernelSpecFactory.class);
        AgentKernelSnapshotBuilder snapshots = mock(AgentKernelSnapshotBuilder.class);
        AgentRunCoordinator coordinator = mock(AgentRunCoordinator.class);
        AgentExecutionRuntimeContextRequests runtimeContexts =
                mock(AgentExecutionRuntimeContextRequests.class);
        AgentExecutionFactory executionFactory = mock(AgentExecutionFactory.class);
        RunExecutionSupervisor supervisor = mock(RunExecutionSupervisor.class);
        AgentRunQueryService queries = mock(AgentRunQueryService.class);
        AgentRunReplayService replay = mock(AgentRunReplayService.class);
        AgentRuntimeInstanceIdentity identity = mock(AgentRuntimeInstanceIdentity.class);
        AgentScopeV2Properties properties = new AgentScopeV2Properties();
        AgentKernelSpec spec = mock(AgentKernelSpec.class);
        AgentScopeRuntimeContextRequest runtime = mock(AgentScopeRuntimeContextRequest.class);
        AgentKernelSnapshot snapshot = snapshot();
        AgentRun previous = AgentRun.builder()
                .runId("failed-run")
                .conversationId("conversation-1")
                .userId(42L)
                .projectId(9L)
                .agentType("ai_assistant_agent")
                .agentStateSessionId("afv:v2:conversation-1:ai_assistant_agent")
                .kernelFingerprint(snapshot.fingerprint())
                .agentDefinitionSnapshotJson(snapshot.snapshotJson())
                .status(status)
                .deadlineAt(LocalDateTime.now().plusMinutes(5))
                .build();

        when(queries.resolveAuthorizedTarget(isNull(), eq("conversation-1"), eq(42L)))
                .thenReturn(Mono.just(previous));
        when(executionFactory.resolve(any(AgentKernelSnapshot.class))).thenReturn(Mono.just(spec));
        when(persistedMessages.listRootHistory("conversation-1")).thenReturn(List.of());
        when(spec.agentDefinitionStableKey()).thenReturn("ai_assistant_agent");
        when(identity.value()).thenReturn("node-2");
        when(coordinator.start(any(StartAgentRunCommand.class)))
                .thenAnswer(invocation -> {
                    StartAgentRunCommand command = invocation.getArgument(0);
                    return Mono.just(new StartedAgentRun(
                            command.runId(),
                            command.conversationId(),
                            command.stateSessionCandidate(),
                            command.ownerInstanceId(),
                            1L,
                            command.deadline().minusSeconds(1),
                            command.deadline(),
                            snapshot,
                            2L));
                });
        when(runtimeContexts.forRoot(
                any(), eq("ai_assistant_agent"), any(),
                eq(ToolExecutionMode.ALWAYS_ASK)))
                .thenReturn(Mono.just(runtime));
        when(supervisor.start(any(StartAgentExecutionCommand.class))).thenReturn(Mono.empty());

        AgentScopePipelineRunService service = new AgentScopePipelineRunService(
                models,
                agents,
                conversations,
                persistedMessages,
                specs,
                snapshots,
                new AgentScopeMessageMapper(),
                coordinator,
                runtimeContexts,
                executionFactory,
                supervisor,
                queries,
                replay,
                identity,
                properties,
                schedulers,
                new ObjectMapper(),
                mock(CanvasAgentScopeService.class));

        StepVerifier.create(service.startContinuation("conversation-1", 42L))
                .assertNext(started -> assertThat(started.conversationId())
                        .isEqualTo("conversation-1"))
                .verifyComplete();

        ArgumentCaptor<StartAgentRunCommand> admission =
                ArgumentCaptor.forClass(StartAgentRunCommand.class);
        verify(coordinator).start(admission.capture());
        assertThat(admission.getValue().userContent()).isEqualTo("继续");
        assertThat(admission.getValue().projectId()).isEqualTo(9L);
        assertThat(admission.getValue().stateSessionCandidate())
                .startsWith("afv-root:")
                .isNotEqualTo(previous.getAgentStateSessionId());
        assertThat(admission.getValue().kernelSnapshot().snapshotJson())
                .isEqualTo(snapshot.snapshotJson());

        ArgumentCaptor<StartAgentExecutionCommand> execution =
                ArgumentCaptor.forClass(StartAgentExecutionCommand.class);
        verify(supervisor).start(execution.capture());
        assertThat(execution.getValue().messages()).singleElement()
                .isInstanceOfSatisfying(UserMessage.class, message ->
                        assertThat(message.getTextContent()).isEqualTo("继续"));
        assertThat(execution.getValue().kernelSpec()).isSameAs(spec);
        verify(persistedMessages).listRootHistory("conversation-1");
        verify(runtimeContexts).forRoot(
                any(), eq("ai_assistant_agent"), any(),
                eq(ToolExecutionMode.ALWAYS_ASK));
    }

    @Test
    void rejectsContinuationWhenLatestRootRunDidNotFail() {
        AgentRunCoordinator coordinator = mock(AgentRunCoordinator.class);
        AgentRunQueryService queries = mock(AgentRunQueryService.class);
        AgentRun previous = AgentRun.builder()
                .runId("completed-run")
                .conversationId("conversation-1")
                .userId(42L)
                .status(AgentRunStatus.COMPLETED.name())
                .deadlineAt(LocalDateTime.now().plusMinutes(5))
                .build();
        when(queries.resolveAuthorizedTarget(isNull(), eq("conversation-1"), eq(42L)))
                .thenReturn(Mono.just(previous));

        AgentScopePipelineRunService service = new AgentScopePipelineRunService(
                mock(AiModelService.class),
                mock(AiAgentService.class),
                mock(AgentConversationService.class),
                mock(AgentMessageService.class),
                mock(AgentKernelSpecFactory.class),
                mock(AgentKernelSnapshotBuilder.class),
                new AgentScopeMessageMapper(),
                coordinator,
                mock(AgentExecutionRuntimeContextRequests.class),
                mock(AgentExecutionFactory.class),
                mock(RunExecutionSupervisor.class),
                queries,
                mock(AgentRunReplayService.class),
                mock(AgentRuntimeInstanceIdentity.class),
                new AgentScopeV2Properties(),
                schedulers,
                new ObjectMapper(),
                mock(CanvasAgentScopeService.class));

        StepVerifier.create(service.startContinuation("conversation-1", 42L))
                .expectErrorSatisfies(error -> assertThat(error)
                        .hasMessage("只有失败或已取消的 Pipeline 可以继续执行"))
                .verify();

        verify(coordinator, never()).start(any(StartAgentRunCommand.class));
    }

    private AgentKernelSnapshot snapshot() {
        return new CanonicalAgentKernelSnapshotBuilder().build(
                new AgentKernelSnapshotPayload(
                        AgentKernelSnapshotPayload.CURRENT_SCHEMA_VERSION,
                        "ai_assistant_agent",
                        "assistant",
                        "test",
                        "system",
                        Map.of(AgentKernelSpecFactory.TOOL_EXECUTION_MODE_VARIABLE, "ALWAYS_ASK"),
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
