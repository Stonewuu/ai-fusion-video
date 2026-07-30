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
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentScopeSkillRegistry;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentUserSkillService;
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
        AgentScopeSkillRegistry skillRegistry = mock(AgentScopeSkillRegistry.class);
        AgentUserSkillService userSkillService = mock(AgentUserSkillService.class);
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
                .supportReasoning(true)
                .reasoningEffortLevels(List.of("high", "low"))
                .build();
        AgentKernelSnapshot snapshot = snapshot();

        when(models.getDefaultByType(1)).thenReturn(model);
        when(skillRegistry.skills()).thenReturn(List.of());
        when(userSkillService.list(42L)).thenReturn(List.of(
                new AgentUserSkillService.UserSkill(
                        "story-review_workspace:user",
                        "story-review",
                        "故事结构检查",
                        "检查故事结构",
                        "# 工作方式\n\n先检查冲突与节奏。",
                        "workspace:user")));
        when(specs.createRoot(any(AiChatReqVO.class), any(AiModel.class), any(String.class), eq(42L)))
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
        when(runtimeContexts.forRoot(
                any(), eq("ai_assistant_agent"), isNull(), eq(ToolExecutionMode.DEFAULT)))
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
                new ObjectMapper(),
                skillRegistry,
                userSkillService);
        AiChatReqVO request = new AiChatReqVO()
                .setConversationId("conversation-1")
                .setMessage("hello harness")
                .setReasoningEffort("high")
                .setToolExecutionMode(ToolExecutionMode.DEFAULT.name())
                .setEnabledSkills(List.of("story-review"));

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

        ArgumentCaptor<String> systemPrompt = ArgumentCaptor.forClass(String.class);
        ArgumentCaptor<AiModel> effectiveModel = ArgumentCaptor.forClass(AiModel.class);
        verify(specs).createRoot(
                any(AiChatReqVO.class), effectiveModel.capture(), systemPrompt.capture(), eq(42L));
        assertThat(effectiveModel.getValue().getConfig()).contains("\"reasoningEffort\":\"high\"");
        assertThat(model.getConfig()).isNull();
        assertThat(systemPrompt.getValue())
                .contains("已主动激活的 Skills", "story-review", "先检查冲突与节奏")
                .doesNotContain("故事结构检查");

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
