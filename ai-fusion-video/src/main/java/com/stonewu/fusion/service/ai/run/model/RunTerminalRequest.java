package com.stonewu.fusion.service.ai.run.model;

import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentRuntimeErrorCode;
import com.stonewu.fusion.enums.ai.AgentTerminalOutputType;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;

import java.util.Objects;
import java.util.Set;

/** Immutable terminal compare-and-set request. */
public record RunTerminalRequest(
        String runId,
        StateStoreSlot stateStoreSlot,
        Set<AgentRunStatus> expectedStatuses,
        AgentRunStatus terminalStatus,
        AgentTerminalOutputType terminalOutputType,
        AgentRuntimeErrorCode errorCode,
        String errorMessage,
        AgentEventEnvelope terminalEnvelope) {

    public RunTerminalRequest {
        requireText(runId, "runId");
        Objects.requireNonNull(stateStoreSlot, "stateStoreSlot must not be null");
        expectedStatuses = Set.copyOf(Objects.requireNonNull(
                expectedStatuses, "expectedStatuses must not be null"));
        if (expectedStatuses.isEmpty()) {
            throw new IllegalArgumentException("expectedStatuses must not be empty");
        }
        if (expectedStatuses.stream().anyMatch(AgentRunStatus::isTerminal)) {
            throw new IllegalArgumentException("expectedStatuses must contain only active states");
        }

        Objects.requireNonNull(terminalStatus, "terminalStatus must not be null");
        if (!terminalStatus.isTerminal()) {
            throw new IllegalArgumentException("terminalStatus must be terminal");
        }
        Objects.requireNonNull(terminalOutputType, "terminalOutputType must not be null");
        if (terminalOutputType.terminalStatus() != terminalStatus) {
            throw new IllegalArgumentException(
                    "terminalOutputType does not match terminalStatus");
        }
        Objects.requireNonNull(terminalEnvelope, "terminalEnvelope must not be null");
        if (!terminalOutputType.name().equals(terminalEnvelope.outputType())) {
            throw new IllegalArgumentException(
                    "terminalEnvelope outputType does not match terminalOutputType");
        }

        if (terminalStatus == AgentRunStatus.FAILED) {
            Objects.requireNonNull(errorCode, "errorCode must not be null for FAILED");
            requireText(errorMessage, "errorMessage");
        } else if (terminalStatus == AgentRunStatus.COMPLETED) {
            if (errorCode != null || errorMessage != null) {
                throw new IllegalArgumentException(
                        "COMPLETED must not carry an error code or message");
            }
        } else if (terminalStatus == AgentRunStatus.CANCELLED) {
            if (errorCode != null && errorCode != AgentRuntimeErrorCode.RUN_CANCELLED) {
                throw new IllegalArgumentException(
                        "CANCELLED may only carry RUN_CANCELLED as its error code");
            }
            if (errorCode == null && errorMessage != null) {
                throw new IllegalArgumentException(
                        "CANCELLED errorMessage requires RUN_CANCELLED");
            }
            if (errorMessage != null) {
                requireText(errorMessage, "errorMessage");
            }
        }
    }

    public RunTerminalRequest asFailure(
            AgentRuntimeErrorCode failureCode,
            String failureMessage,
            AgentEventEnvelope failureEnvelope) {
        return new RunTerminalRequest(
                runId,
                stateStoreSlot,
                expectedStatuses,
                AgentRunStatus.FAILED,
                AgentTerminalOutputType.ERROR,
                Objects.requireNonNull(failureCode, "failureCode must not be null"),
                failureMessage,
                Objects.requireNonNull(failureEnvelope, "failureEnvelope must not be null"));
    }

    private static void requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
    }
}
