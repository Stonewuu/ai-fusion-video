package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.controller.ai.vo.ToolConfirmationReqVO;
import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextRequest;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.PendingConfirmation;
import com.stonewu.fusion.service.ai.run.model.ResumeAgentExecutionCommand;
import com.stonewu.fusion.service.ai.run.model.ResumeConfirmationCommand;
import com.stonewu.fusion.service.ai.run.model.ResumedAgentRun;
import io.agentscope.core.event.ConfirmResult;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.ToolCallState;
import io.agentscope.core.message.ToolUseBlock;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.mockito.ArgumentMatchers.any;

class AgentConfirmationServiceTests {

    private final ObjectMapper objectMapper = new ObjectMapper().findAndRegisterModules();

    @Test
    void deniedDecisionIsReplayedAsAgentScopeConfirmResult() throws Exception {
        AgentWaitingStatePort waiting = mock(AgentWaitingStatePort.class);
        AgentExecutionRuntimeContextRequests runtimeContexts =
                mock(AgentExecutionRuntimeContextRequests.class);
        RunExecutionSupervisor supervisor = mock(RunExecutionSupervisor.class);
        AgentScopeV2Properties properties = new AgentScopeV2Properties();
        properties.getExecution().setInstanceId("confirm-node");
        AgentRuntimeInstanceIdentity identity = new AgentRuntimeInstanceIdentity(properties);
        CanonicalAgentKernelSnapshotBuilder snapshots =
                new CanonicalAgentKernelSnapshotBuilder(objectMapper);
        AgentConfirmationService service = new AgentConfirmationService(
                waiting,
                runtimeContexts,
                supervisor,
                snapshots,
                identity,
                properties,
                objectMapper);

        ToolUseBlock toolCall = new ToolUseBlock(
                "call-1",
                "update_script",
                Map.of("scriptId", 7),
                null,
                Map.of(),
                ToolCallState.ASKING);
        PendingConfirmation pending = new PendingConfirmation(
                "reply-1",
                Set.of("call-1"),
                "[{\"toolCallId\":\"call-1\",\"toolName\":\"update_script\",\"argumentsPreview\":\"{}\"}]",
                objectMapper.writeValueAsString(List.of(Map.of(
                        "id", toolCall.getId(),
                        "name", toolCall.getName(),
                        "input", toolCall.getInput(),
                        "metadata", toolCall.getMetadata(),
                        "state", toolCall.getState().name()))),
                Instant.now().plusSeconds(300));
        AgentKernelSnapshot snapshot = snapshots.build(new AgentKernelSnapshotPayload(
                AgentKernelSnapshotPayload.CURRENT_SCHEMA_VERSION,
                "assistant",
                "assistant",
                "test assistant",
                "system",
                Map.of(
                        AgentKernelSpecFactory.TOOL_EXECUTION_MODE_VARIABLE,
                        ToolExecutionMode.DEFAULT.name()),
                5,
                "1",
                1,
                "openai",
                "test-model",
                JsonNodeFactory.instance.objectNode(),
                List.of(),
                "test"));
        ResumedAgentRun resumed = new ResumedAgentRun(
                "run-1",
                "conversation-1",
                "session-1",
                snapshot.fingerprint(),
                snapshot.snapshotJson(),
                3,
                "confirm-node",
                2,
                Instant.now().plusSeconds(30),
                Instant.now().plusSeconds(300));
        AgentScopeRuntimeContextRequest runtime = mock(AgentScopeRuntimeContextRequest.class);
        when(waiting.getPendingConfirmationAuthorized("7", 42, "reply-1"))
                .thenReturn(Mono.just(pending));
        when(waiting.resumeConfirmation(any()))
                .thenReturn(Mono.just(resumed));
        when(runtimeContexts.forResume(
                resumed, "assistant", ToolExecutionMode.DEFAULT))
                .thenReturn(Mono.just(runtime));
        when(supervisor.resume(any()))
                .thenReturn(Mono.empty());

        ToolConfirmationReqVO request = request(false);
        StepVerifier.create(service.respond(request, 42)).verifyComplete();

        ArgumentCaptor<ResumeConfirmationCommand> transition =
                ArgumentCaptor.forClass(ResumeConfirmationCommand.class);
        verify(waiting).resumeConfirmation(transition.capture());
        assertThat(transition.getValue().decisionResults())
                .containsEntry("call-1", false);

        ArgumentCaptor<ResumeAgentExecutionCommand> execution =
                ArgumentCaptor.forClass(ResumeAgentExecutionCommand.class);
        verify(supervisor).resume(execution.capture());
        Msg resumeMessage = execution.getValue().messages().getFirst();
        Object metadata = resumeMessage.getMetadata().get(Msg.METADATA_CONFIRM_RESULTS);
        assertThat(metadata).isInstanceOf(List.class);
        ConfirmResult result = (ConfirmResult) ((List<?>) metadata).getFirst();
        assertThat(result.isConfirmed()).isFalse();
        assertThat(result.getToolCall().getId()).isEqualTo("call-1");
    }

    private ToolConfirmationReqVO request(boolean approved) {
        ToolConfirmationReqVO.DecisionVO decision = new ToolConfirmationReqVO.DecisionVO();
        decision.setToolCallId("call-1");
        decision.setApproved(approved);
        ToolConfirmationReqVO request = new ToolConfirmationReqVO();
        request.setRunId("7");
        request.setReplyId("reply-1");
        request.setDecisions(List.of(decision));
        return request;
    }
}
