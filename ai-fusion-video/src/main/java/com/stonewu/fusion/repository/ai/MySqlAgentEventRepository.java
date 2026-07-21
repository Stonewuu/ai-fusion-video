package com.stonewu.fusion.repository.ai;

import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentModelCallStatus;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentRuntimeErrorCode;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentModelCallUsageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.run.AgentEventEnvelopeSanitizer;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.SystemTerminalActor;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/** MySQL row-lock implementation of the durable event transaction port. */
@Repository
@RequiredArgsConstructor
public class MySqlAgentEventRepository implements AgentEventRepository {

    private static final Set<AgentRunStatus> OWNED_TERMINAL_SOURCE =
            Set.of(AgentRunStatus.RUNNING);

    private final AgentRunMapper runMapper;
    private final AgentEventMapper eventMapper;
    private final AgentModelCallUsageMapper usageMapper;
    private final ObjectMapper objectMapper;
    private final AgentEventEnvelopeSanitizer sanitizer;

    @Override
    @Transactional
    public Optional<CommittedAgentEvent> appendOwnedTx(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            AgentEventEnvelope event) {
        requireOwnerArguments(runId, ownerInstanceId, ownerEpoch);
        Objects.requireNonNull(event, "event must not be null");

        AgentRun run = runMapper.selectByRunIdForUpdate(runId);
        LocalDateTime databaseNow = runMapper.selectDatabaseNow();
        if (!isCurrentOwner(run, ownerInstanceId, ownerEpoch, databaseNow)) {
            return Optional.empty();
        }
        AgentEventEnvelope safeEvent = sanitize(event);
        long sequence = requireNextSequence(run);
        advanceSequence(run, sequence + 1, databaseNow);
        AgentEvent inserted = insertEvent(run, sequence, safeEvent, databaseNow);
        return Optional.of(toCommitted(inserted, safeEvent, databaseNow));
    }

    @Override
    @Transactional
    public Optional<CommittedAgentEvent> terminateOwnedTx(
            RunTerminalRequest request,
            String ownerInstanceId,
            long ownerEpoch) {
        Objects.requireNonNull(request, "request must not be null");
        requireOwnerArguments(request.runId(), ownerInstanceId, ownerEpoch);

        AgentRun run = runMapper.selectByRunIdForUpdate(request.runId());
        LocalDateTime databaseNow = runMapper.selectDatabaseNow();
        if (!OWNED_TERMINAL_SOURCE.equals(request.expectedStatuses())
                || !isCurrentOwner(run, ownerInstanceId, ownerEpoch, databaseNow)) {
            return Optional.empty();
        }
        return Optional.of(insertTerminal(run, request, databaseNow));
    }

    @Override
    @Transactional
    public Optional<CommittedAgentEvent> terminateSystemTx(
            RunTerminalRequest request,
            SystemTerminalActor actor) {
        Objects.requireNonNull(request, "request must not be null");
        Objects.requireNonNull(actor, "actor must not be null");

        AgentRun run = runMapper.selectByRunIdForUpdate(request.runId());
        LocalDateTime databaseNow = runMapper.selectDatabaseNow();
        if (!systemTransitionAllowed(run, request, actor, databaseNow)) {
            return Optional.empty();
        }
        return Optional.of(insertTerminal(run, request, databaseNow));
    }

    private boolean isCurrentOwner(
            AgentRun run,
            String ownerInstanceId,
            long ownerEpoch,
            LocalDateTime databaseNow) {
        return run != null
                && AgentRunStatus.RUNNING.name().equals(run.getStatus())
                && Objects.equals(run.getOwnerInstanceId(), ownerInstanceId)
                && Objects.equals(run.getOwnerEpoch(), ownerEpoch)
                && run.getLeaseUntil() != null
                && run.getLeaseUntil().isAfter(databaseNow);
    }

    private boolean systemTransitionAllowed(
            AgentRun run,
            RunTerminalRequest request,
            SystemTerminalActor actor,
            LocalDateTime databaseNow) {
        if (run == null) {
            return false;
        }
        AgentRunStatus current = parseStatus(run.getStatus());
        if (!request.expectedStatuses().equals(Set.of(current))) {
            return false;
        }
        return switch (actor) {
            case CANCELLATION_COORDINATOR ->
                    current == AgentRunStatus.CANCEL_REQUESTED
                            && request.terminalStatus() == AgentRunStatus.CANCELLED;
            case OWNER_RECONCILER ->
                    run.getLeaseUntil() != null
                            && !run.getLeaseUntil().isAfter(databaseNow)
                            && ((current == AgentRunStatus.RUNNING
                                    && request.terminalStatus() == AgentRunStatus.FAILED
                                    && request.errorCode() == AgentRuntimeErrorCode.OWNER_LOST)
                                || (current == AgentRunStatus.CANCEL_REQUESTED
                                    && request.terminalStatus() == AgentRunStatus.CANCELLED));
        };
    }

    private CommittedAgentEvent insertTerminal(
            AgentRun run,
            RunTerminalRequest request,
            LocalDateTime databaseNow) {
        AgentEventEnvelope safeEnvelope = sanitize(request.terminalEnvelope());
        validateTerminalIdentity(run, safeEnvelope);
        long sequence = requireNextSequence(run);
        AgentEvent inserted = insertEvent(
                run, sequence, safeEnvelope, databaseNow);

        int updated = runMapper.update(null, new LambdaUpdateWrapper<AgentRun>()
                .eq(AgentRun::getId, run.getId())
                .eq(AgentRun::getStatus, run.getStatus())
                .set(AgentRun::getNextSequence, sequence + 1)
                .set(AgentRun::getTerminalSequence, sequence)
                .set(AgentRun::getTerminalOutputType,
                        request.terminalOutputType().getCode())
                .set(AgentRun::getStatus, request.terminalStatus().name())
                .set(AgentRun::getFinishedAt, databaseNow)
                .set(AgentRun::getErrorCode,
                        request.errorCode() == null ? null : request.errorCode().getCode())
                .set(AgentRun::getErrorMessage, request.errorMessage())
                .set(AgentRun::getWaitingReplyId, null)
                .set(AgentRun::getWaitingToolCallId, null)
                .set(AgentRun::getWaitingToolName, null)
                .set(AgentRun::getWaitExpiresAt, null)
                .set(AgentRun::getUpdateTime, databaseNow));
        if (updated != 1) {
            throw new IllegalStateException(
                    "Agent terminal update did not affect exactly one row");
        }
        usageMapper.finishAllStartedForRun(
                run.getRunId(),
                request.terminalStatus() == AgentRunStatus.CANCELLED
                        ? AgentModelCallStatus.CANCELLED.name()
                        : AgentModelCallStatus.FAILED.name(),
                databaseNow);
        return toCommitted(inserted, safeEnvelope, databaseNow);
    }

    private AgentEvent insertEvent(
            AgentRun run,
            long sequence,
            AgentEventEnvelope envelope,
            LocalDateTime databaseNow) {
        boolean publishRequired = envelope.outputType() != null;
        AgentEvent row = AgentEvent.builder()
                .runId(run.getRunId())
                .sequenceNo(sequence)
                .schemaVersion(1)
                .rawEventId(envelope.rawEventId())
                .rawEventType(envelope.rawEventType())
                .source(envelope.source())
                .replyId(envelope.replyId())
                .blockId(envelope.blockId())
                .toolCallId(envelope.toolCallId())
                .parentToolCallId(envelope.parentToolCallId())
                .agentName(envelope.agentName())
                .outputType(envelope.outputType())
                .payloadJson(envelope.payloadJson(objectMapper))
                .eventCreatedAt(toDatabaseTime(envelope.createdAt()))
                .publishRequired(publishRequired)
                .publishStatus(publishRequired ? "PENDING" : "NOT_REQUIRED")
                .nextPublishAttemptAt(publishRequired ? databaseNow : null)
                .build();
        if (eventMapper.insert(row) != 1) {
            throw new IllegalStateException(
                    "Agent event insert did not affect exactly one row");
        }
        return row;
    }

    private void advanceSequence(
            AgentRun run, long nextSequence, LocalDateTime databaseNow) {
        int updated = runMapper.update(null, new LambdaUpdateWrapper<AgentRun>()
                .eq(AgentRun::getId, run.getId())
                .eq(AgentRun::getStatus, AgentRunStatus.RUNNING.name())
                .set(AgentRun::getNextSequence, nextSequence)
                .set(AgentRun::getUpdateTime, databaseNow));
        if (updated != 1) {
            throw new IllegalStateException(
                    "Agent sequence update did not affect exactly one row");
        }
    }

    private void validateTerminalIdentity(AgentRun run, AgentEventEnvelope envelope) {
        if (run.getParentRunId() == null) {
            if (envelope.parentToolCallId() != null || envelope.agentName() != null) {
                throw new IllegalArgumentException(
                        "Root terminal envelope must not carry child identity");
            }
        } else if (!Objects.equals(run.getParentToolCallId(), envelope.parentToolCallId())
                || !Objects.equals(run.getAgentName(), envelope.agentName())) {
            throw new IllegalArgumentException(
                    "Child terminal envelope identity does not match the run");
        }
    }

    private CommittedAgentEvent toCommitted(
            AgentEvent row,
            AgentEventEnvelope envelope,
            LocalDateTime databaseNow) {
        return new CommittedAgentEvent(
                row.getId(), row.getRunId(), row.getSequenceNo(), envelope,
                databaseNow.toInstant(ZoneOffset.UTC));
    }

    private long requireNextSequence(AgentRun run) {
        Long sequence = run.getNextSequence();
        if (sequence == null || sequence < 1) {
            throw new IllegalStateException(
                    "Agent run has an invalid next sequence: " + run.getRunId());
        }
        return sequence;
    }

    private AgentRunStatus parseStatus(String status) {
        try {
            return AgentRunStatus.valueOf(status);
        } catch (RuntimeException invalid) {
            throw new IllegalStateException("Agent run has an invalid status: " + status, invalid);
        }
    }

    private void requireOwnerArguments(
            String runId, String ownerInstanceId, long ownerEpoch) {
        if (runId == null || runId.isBlank()) {
            throw new IllegalArgumentException("runId must not be blank");
        }
        if (ownerInstanceId == null || ownerInstanceId.isBlank()) {
            throw new IllegalArgumentException("ownerInstanceId must not be blank");
        }
        if (ownerEpoch <= 0) {
            throw new IllegalArgumentException("ownerEpoch must be positive");
        }
    }

    private AgentEventEnvelope sanitize(AgentEventEnvelope envelope) {
        return new AgentEventEnvelope(
                envelope.rawEventId(),
                envelope.rawEventType(),
                envelope.source(),
                envelope.replyId(),
                envelope.blockId(),
                envelope.toolCallId(),
                envelope.parentToolCallId(),
                envelope.agentName(),
                envelope.outputType(),
                sanitizer.sanitize(envelope.payload()),
                envelope.createdAt());
    }

    private LocalDateTime toDatabaseTime(Instant instant) {
        return LocalDateTime.ofInstant(instant, ZoneOffset.UTC);
    }
}
