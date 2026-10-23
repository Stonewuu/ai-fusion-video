package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.controller.ai.vo.ToolConfirmationReqVO;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.model.PendingConfirmation;
import com.stonewu.fusion.service.ai.run.model.ResumeAgentExecutionCommand;
import com.stonewu.fusion.service.ai.run.model.ResumeConfirmationCommand;
import com.stonewu.fusion.service.ai.run.model.ResumedAgentRun;
import io.agentscope.core.message.Msg;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.util.*;

/**
 * Authorizes, durably claims, and resumes a user-confirmed AgentScope tool pause.
 */
@Service
public final class AgentConfirmationService {

    private final AgentWaitingStatePort waitingState;
    private final AgentExecutionRuntimeContextRequests runtimeContexts;
    private final RunExecutionSupervisor supervisor;
    private final AgentKernelSnapshotBuilder snapshotBuilder;
    private final AgentRuntimeInstanceIdentity instanceIdentity;
    private final AgentScopeV2Properties properties;
    private final ObjectMapper objectMapper;

    public AgentConfirmationService(
            AgentWaitingStatePort waitingState,
            AgentExecutionRuntimeContextRequests runtimeContexts,
            RunExecutionSupervisor supervisor,
            AgentKernelSnapshotBuilder snapshotBuilder,
            AgentRuntimeInstanceIdentity instanceIdentity,
            AgentScopeV2Properties properties,
            ObjectMapper objectMapper) {
        this.waitingState = Objects.requireNonNull(waitingState, "waitingState must not be null");
        this.runtimeContexts = Objects.requireNonNull(
                runtimeContexts, "runtimeContexts must not be null");
        this.supervisor = Objects.requireNonNull(supervisor, "supervisor must not be null");
        this.snapshotBuilder = Objects.requireNonNull(
                snapshotBuilder, "snapshotBuilder must not be null");
        this.instanceIdentity = Objects.requireNonNull(
                instanceIdentity, "instanceIdentity must not be null");
        this.properties = Objects.requireNonNull(properties, "properties must not be null");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
    }

    public Mono<Void> respond(ToolConfirmationReqVO request, long currentUserId) {
        ToolConfirmationReqVO safeRequest = Objects.requireNonNull(
                request, "request must not be null");
        String runId = requireText(safeRequest.getRunId(), "runId");
        String replyId = requireText(safeRequest.getReplyId(), "replyId");
        Map<String, Boolean> decisions = decisions(safeRequest.getDecisions());

        return waitingState.getPendingConfirmationAuthorized(
                        runId, currentUserId, replyId)
                .flatMap(pending -> resume(
                        pending, decisions, runId, replyId, currentUserId));
    }

    private Mono<Void> resume(
            PendingConfirmation pending,
            Map<String, Boolean> decisions,
            String runId,
            String replyId,
            long currentUserId) {
        if (!pending.decisionIds().equals(decisions.keySet())) {
            return Mono.error(new BusinessException(
                    409, "确认决定与当前待审批工具不一致"));
        }
        Msg resumeMessage = AgentConfirmationMessages.restore(pending.suspendedToolCallsJson(), decisions,
                objectMapper);
        ResumeConfirmationCommand transition = new ResumeConfirmationCommand(
                runId,
                currentUserId,
                replyId,
                Set.copyOf(decisions.keySet()),
                decisions,
                instanceIdentity.value(),
                properties.getExecution().getOwnerLease());
        return waitingState.resumeConfirmation(transition)
                .flatMap(resumed -> launchResume(resumed, resumeMessage));
    }

    private Mono<Void> launchResume(ResumedAgentRun resumed, Msg resumeMessage) {
        if (resumed.durableDispatch()) {
            return Mono.empty();
        }
        AgentKernelSnapshot snapshot = snapshot(resumed);
        ToolExecutionMode toolExecutionMode = ToolExecutionMode.parse(
                snapshot.payload().promptVariables().get(
                        AgentKernelSpecFactory.TOOL_EXECUTION_MODE_VARIABLE));
        return runtimeContexts.forResume(
                        resumed,
                        snapshot.payload().agentDefinitionStableKey(),
                        toolExecutionMode)
                .flatMap(runtime -> supervisor.resume(new ResumeAgentExecutionCommand(
                        resumed, List.of(resumeMessage), snapshot, runtime)));
    }

    private AgentKernelSnapshot snapshot(ResumedAgentRun resumed) {
        try {
            JsonNode root = objectMapper.readTree(resumed.agentDefinitionSnapshotJson());
            AgentKernelSnapshotPayload payload = objectMapper.treeToValue(
                    root, AgentKernelSnapshotPayload.class);
            AgentKernelSnapshot snapshot = snapshotBuilder.build(payload);
            if (!snapshot.fingerprint().equals(resumed.kernelFingerprint())) {
                throw new IllegalStateException(
                        "Persisted resume snapshot fingerprint does not match its payload");
            }
            return snapshot;
        } catch (JsonProcessingException invalidSnapshot) {
            throw new IllegalStateException(
                    "Persisted resume snapshot is invalid", invalidSnapshot);
        }
    }

    private Map<String, Boolean> decisions(List<ToolConfirmationReqVO.DecisionVO> values) {
        if (values == null || values.isEmpty()) {
            throw new BusinessException(400, "确认决定不能为空");
        }
        Map<String, Boolean> result = new LinkedHashMap<>();
        for (ToolConfirmationReqVO.DecisionVO value : values) {
            if (value == null || value.getApproved() == null) {
                throw new BusinessException(400, "每个工具都必须明确允许或拒绝");
            }
            String toolCallId = requireText(value.getToolCallId(), "toolCallId");
            if (result.putIfAbsent(toolCallId, value.getApproved()) != null) {
                throw new BusinessException(400, "工具确认决定不能重复");
            }
        }
        return Map.copyOf(result);
    }

    private String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new BusinessException(400, field + " 不能为空");
        }
        return value.trim();
    }
}
