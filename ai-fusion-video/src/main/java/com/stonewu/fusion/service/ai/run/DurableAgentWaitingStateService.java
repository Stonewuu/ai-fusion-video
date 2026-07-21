package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.PendingConfirmation;
import com.stonewu.fusion.service.ai.run.model.PendingExternalExecution;
import com.stonewu.fusion.service.ai.run.model.ResumeConfirmationCommand;
import com.stonewu.fusion.service.ai.run.model.ResumeExternalCommand;
import com.stonewu.fusion.service.ai.run.model.ResumedAgentRun;
import com.stonewu.fusion.service.ai.run.model.WaitingCheckpoint;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

/** MySQL-backed, owner-fenced WAITING state transitions. */
@Service
public final class DurableAgentWaitingStateService implements AgentWaitingStatePort {

    private static final int MAX_JSON_BYTES = 8 * 1024 * 1024;
    private static final String CANDIDATE_SOURCE = "platform/waiting-candidate";
    private static final String WAITING_SOURCE = "platform/waiting";

    private final AgentRunMapper runMapper;
    private final AgentEventMapper eventMapper;
    private final ObjectMapper objectMapper;
    private final AgentEventEnvelopeSanitizer sanitizer;
    private final TransactionTemplate transactions;
    private final AgentRuntimeSchedulers schedulers;

    public DurableAgentWaitingStateService(
            AgentRunMapper runMapper,
            AgentEventMapper eventMapper,
            ObjectMapper objectMapper,
            AgentEventEnvelopeSanitizer sanitizer,
            TransactionTemplate transactions,
            AgentRuntimeSchedulers schedulers) {
        this.runMapper = Objects.requireNonNull(runMapper, "runMapper must not be null");
        this.eventMapper = Objects.requireNonNull(
                eventMapper, "eventMapper must not be null");
        this.objectMapper = Objects.requireNonNull(
                objectMapper, "objectMapper must not be null");
        this.sanitizer = Objects.requireNonNull(sanitizer, "sanitizer must not be null");
        this.transactions = Objects.requireNonNull(
                transactions, "transactions must not be null");
        this.schedulers = Objects.requireNonNull(
                schedulers, "schedulers must not be null");
    }

    @Override
    public Mono<Void> recordConfirmationCandidate(
            String runId, PendingConfirmation candidate) {
        String safeRunId = requireText(runId, 64, "runId");
        PendingConfirmation safeCandidate = Objects.requireNonNull(
                candidate, "candidate must not be null");
        validateConfirmation(safeCandidate);
        return Mono.fromRunnable(() -> transactions.executeWithoutResult(ignored ->
                        recordConfirmationCandidateTx(safeRunId, safeCandidate)))
                .subscribeOn(schedulers.journal())
                .then();
    }

    @Override
    public Mono<Boolean> enterWaitingConfirmation(
            String runId, long expectedOwnerEpoch, WaitingCheckpoint checkpoint) {
        String safeRunId = requireText(runId, 64, "runId");
        requirePositive(expectedOwnerEpoch, "expectedOwnerEpoch");
        WaitingCheckpoint safeCheckpoint = Objects.requireNonNull(
                checkpoint, "checkpoint must not be null");
        return transactional(() -> enterWaitingConfirmationTx(
                safeRunId, expectedOwnerEpoch, safeCheckpoint));
    }

    @Override
    public Mono<Boolean> enterWaitingExternal(
            String runId,
            long expectedOwnerEpoch,
            WaitingCheckpoint checkpoint,
            PendingExternalExecution pending) {
        String safeRunId = requireText(runId, 64, "runId");
        requirePositive(expectedOwnerEpoch, "expectedOwnerEpoch");
        WaitingCheckpoint safeCheckpoint = Objects.requireNonNull(
                checkpoint, "checkpoint must not be null");
        PendingExternalExecution safePending = Objects.requireNonNull(
                pending, "pending must not be null");
        validateExternal(safePending);
        return transactional(() -> enterWaitingExternalTx(
                safeRunId, expectedOwnerEpoch, safeCheckpoint, safePending));
    }

    @Override
    public Mono<PendingConfirmation> getPendingConfirmationAuthorized(
            String runId, long currentUserId, String replyId) {
        String safeRunId = requireText(runId, 64, "runId");
        requirePositive(currentUserId, "currentUserId");
        String safeReplyId = requireText(replyId, 128, "replyId");
        return transactional(() -> getPendingConfirmationTx(
                safeRunId, currentUserId, safeReplyId));
    }

    @Override
    public Mono<PendingExternalExecution> getPendingExternalAuthorized(
            String runId, long currentUserId, String toolCallId) {
        String safeRunId = requireText(runId, 64, "runId");
        requirePositive(currentUserId, "currentUserId");
        String safeToolCallId = requireText(toolCallId, 128, "toolCallId");
        return transactional(() -> getPendingExternalTx(
                safeRunId, currentUserId, safeToolCallId));
    }

    @Override
    public Mono<ResumedAgentRun> resumeConfirmation(ResumeConfirmationCommand command) {
        ResumeConfirmationCommand safeCommand = Objects.requireNonNull(
                command, "command must not be null");
        requireText(safeCommand.runId(), 64, "runId");
        requireText(safeCommand.replyId(), 128, "replyId");
        requireText(safeCommand.newOwnerInstanceId(), 128, "newOwnerInstanceId");
        return transactional(() -> resumeConfirmationTx(safeCommand));
    }

    @Override
    public Mono<ResumedAgentRun> resumeExternal(ResumeExternalCommand command) {
        ResumeExternalCommand safeCommand = Objects.requireNonNull(
                command, "command must not be null");
        requireText(safeCommand.runId(), 64, "runId");
        requireText(safeCommand.internalExecutorId(), 128, "internalExecutorId");
        requireText(safeCommand.toolCallId(), 128, "toolCallId");
        requireText(safeCommand.newOwnerInstanceId(), 128, "newOwnerInstanceId");
        parseObject(safeCommand.resultPayloadJson(), "resultPayloadJson");
        return transactional(() -> resumeExternalTx(safeCommand));
    }

    private void recordConfirmationCandidateTx(
            String runId, PendingConfirmation candidate) {
        AgentRun run = requireRunForUpdate(runId);
        LocalDateTime now = runMapper.selectDatabaseNow();
        if (!isCurrentRunningOwner(run, run.getOwnerEpoch(), now)) {
            throw conflict("Agent 运行已不接受确认候选");
        }
        PendingConfirmation normalized = normalize(candidate, run, now);
        AgentEvent existing = eventMapper.selectConfirmationCandidate(
                runId, normalized.replyId());
        if (existing != null) {
            PendingConfirmation persisted = confirmation(existing);
            if (sameConfirmation(persisted, normalized)) {
                return;
            }
            throw conflict("同一 replyId 已存在不同的确认候选");
        }

        long sequence = nextSequence(run);
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("replyId", normalized.replyId())
                .put("pendingToolCallsJson", normalized.pendingToolCallsJson())
                .put("expiresAt", normalized.expiresAt().toString());
        payload.set("decisionIds", textArray(normalized.decisionIds()));
        insertEvent(
                run,
                sequence,
                "REQUIRE_USER_CONFIRM",
                CANDIDATE_SOURCE,
                normalized.replyId(),
                null,
                null,
                payload,
                now);
        if (runMapper.advanceRunningSequence(
                run.getId(), run.getOwnerEpoch(), increment(sequence), now) != 1) {
            throw conflict("Agent 运行所有权已变更");
        }
    }

    private boolean enterWaitingConfirmationTx(
            String runId, long expectedOwnerEpoch, WaitingCheckpoint checkpoint) {
        AgentRun run = requireRunForUpdate(runId);
        LocalDateTime now = runMapper.selectDatabaseNow();
        if (!isCurrentRunningOwner(run, expectedOwnerEpoch, now)) {
            return false;
        }
        requireCheckpoint(run, checkpoint);
        AgentEvent candidateEvent = eventMapper.selectLatestConfirmationCandidate(runId);
        if (candidateEvent == null) {
            throw conflict("Agent 运行没有可恢复的确认候选");
        }
        PendingConfirmation candidate = confirmation(candidateEvent);
        LocalDateTime expiresAt = requireActionableExpiry(
                candidate.expiresAt(), run, now);
        long sequence = nextSequence(run);
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", "TOOL_CALL")
                .put("controlType", "USER_CONFIRM_REQUIRED")
                .put("replyId", candidate.replyId())
                .put("toolName", "user_confirmation")
                .put("arguments", candidate.pendingToolCallsJson())
                .put("expiresAt", toInstant(expiresAt).toString());
        payload.set("pendingToolCalls", parseArray(
                candidate.pendingToolCallsJson(), "pendingToolCallsJson"));
        insertEvent(
                run,
                sequence,
                "PLATFORM_USER_CONFIRM_REQUIRED",
                WAITING_SOURCE,
                candidate.replyId(),
                candidate.replyId(),
                "TOOL_CALL",
                sanitizer.sanitize(payload),
                now);
        int updated = runMapper.enterWaitingConfirmation(
                run.getId(),
                expectedOwnerEpoch,
                candidate.replyId(),
                expiresAt,
                checkpoint.pausedThroughSequence(),
                increment(sequence),
                now);
        if (updated != 1) {
            throw conflict("Agent 运行所有权已变更");
        }
        return true;
    }

    private boolean enterWaitingExternalTx(
            String runId,
            long expectedOwnerEpoch,
            WaitingCheckpoint checkpoint,
            PendingExternalExecution pending) {
        AgentRun run = requireRunForUpdate(runId);
        LocalDateTime now = runMapper.selectDatabaseNow();
        if (!isCurrentRunningOwner(run, expectedOwnerEpoch, now)) {
            return false;
        }
        requireCheckpoint(run, checkpoint);
        LocalDateTime expiresAt = clampedExpiry(pending.expiresAt(), run, now);
        PendingExternalExecution normalized = new PendingExternalExecution(
                pending.toolCallId(),
                pending.toolName(),
                pending.suspendedPayloadJson(),
                toInstant(expiresAt));
        AgentEvent existing = eventMapper.selectPendingExternalExecution(
                runId, pending.toolCallId());
        if (existing != null && !sameExternal(external(existing), normalized)) {
            throw conflict("同一 toolCallId 已存在不同的外部执行候选");
        }
        if (existing != null) {
            throw conflict("外部执行候选已处理");
        }

        long sequence = nextSequence(run);
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("toolCallId", normalized.toolCallId())
                .put("toolName", normalized.toolName())
                .put("suspendedPayloadJson", normalized.suspendedPayloadJson())
                .put("expiresAt", normalized.expiresAt().toString());
        insertEvent(
                run,
                sequence,
                "PLATFORM_REQUIRE_EXTERNAL_EXECUTION",
                WAITING_SOURCE,
                null,
                normalized.toolCallId(),
                null,
                payload,
                now);
        int updated = runMapper.enterWaitingExternal(
                run.getId(),
                expectedOwnerEpoch,
                normalized.toolCallId(),
                normalized.toolName(),
                expiresAt,
                checkpoint.pausedThroughSequence(),
                increment(sequence),
                now);
        if (updated != 1) {
            throw conflict("Agent 运行所有权已变更");
        }
        return true;
    }

    private PendingConfirmation getPendingConfirmationTx(
            String runId, long currentUserId, String replyId) {
        AgentRun run = requireAuthorizedRun(runId, currentUserId);
        LocalDateTime now = runMapper.selectDatabaseNow();
        requireWaiting(run, AgentRunStatus.WAITING_CONFIRMATION, now);
        if (!Objects.equals(run.getWaitingReplyId(), replyId)) {
            throw conflict("确认请求已处理或 replyId 不匹配");
        }
        AgentEvent event = eventMapper.selectConfirmationCandidate(runId, replyId);
        if (event == null) {
            throw conflict("确认候选不存在");
        }
        PendingConfirmation pending = confirmation(event);
        requireSameExpiry(run, pending.expiresAt());
        return pending;
    }

    private PendingExternalExecution getPendingExternalTx(
            String runId, long currentUserId, String toolCallId) {
        AgentRun run = requireAuthorizedRun(runId, currentUserId);
        LocalDateTime now = runMapper.selectDatabaseNow();
        requireWaiting(run, AgentRunStatus.WAITING_EXTERNAL, now);
        if (!Objects.equals(run.getWaitingToolCallId(), toolCallId)) {
            throw conflict("外部执行请求已处理或 toolCallId 不匹配");
        }
        AgentEvent event = eventMapper.selectPendingExternalExecution(runId, toolCallId);
        if (event == null) {
            throw conflict("外部执行候选不存在");
        }
        PendingExternalExecution pending = external(event);
        if (!Objects.equals(run.getWaitingToolName(), pending.toolName())) {
            throw conflict("外部执行工具身份不匹配");
        }
        requireSameExpiry(run, pending.expiresAt());
        return pending;
    }

    private ResumedAgentRun resumeConfirmationTx(ResumeConfirmationCommand command) {
        AgentRun run = requireAuthorizedRun(command.runId(), command.currentUserId());
        LocalDateTime now = runMapper.selectDatabaseNow();
        requireWaiting(run, AgentRunStatus.WAITING_CONFIRMATION, now);
        if (!Objects.equals(run.getWaitingReplyId(), command.replyId())) {
            throw conflict("确认请求已处理或 replyId 不匹配");
        }
        AgentEvent candidateEvent = eventMapper.selectConfirmationCandidate(
                command.runId(), command.replyId());
        if (candidateEvent == null) {
            throw conflict("确认候选不存在");
        }
        PendingConfirmation pending = confirmation(candidateEvent);
        if (!pending.decisionIds().equals(command.decisionIds())) {
            throw conflict("确认决定集合与待确认工具不一致");
        }
        requireSameExpiry(run, pending.expiresAt());

        long newEpoch = incrementOwnerEpoch(run);
        LocalDateTime leaseUntil = leaseUntil(now, command.ownerLease(), run.getDeadlineAt());
        long sequence = nextSequence(run);
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("replyId", command.replyId())
                .put("ownerInstanceId", command.newOwnerInstanceId())
                .put("ownerEpoch", newEpoch);
        payload.set("decisionIds", textArray(command.decisionIds()));
        insertEvent(
                run,
                sequence,
                "USER_CONFIRM_RESULT",
                WAITING_SOURCE,
                command.replyId(),
                null,
                null,
                payload,
                now);
        if (runMapper.resumeConfirmation(
                run.getId(),
                run.getOwnerEpoch(),
                command.replyId(),
                command.newOwnerInstanceId(),
                newEpoch,
                leaseUntil,
                increment(sequence),
                now) != 1) {
            throw conflict("确认请求已由其他节点处理");
        }
        return resumed(run, command.newOwnerInstanceId(), newEpoch, leaseUntil);
    }

    private ResumedAgentRun resumeExternalTx(ResumeExternalCommand command) {
        AgentRun run = requireAuthorizedRun(command.runId(), command.currentUserId());
        LocalDateTime now = runMapper.selectDatabaseNow();
        requireWaiting(run, AgentRunStatus.WAITING_EXTERNAL, now);
        if (!Objects.equals(run.getWaitingToolCallId(), command.toolCallId())) {
            throw conflict("外部执行请求已处理或 toolCallId 不匹配");
        }
        AgentEvent pendingEvent = eventMapper.selectPendingExternalExecution(
                command.runId(), command.toolCallId());
        if (pendingEvent == null) {
            throw conflict("外部执行候选不存在");
        }
        PendingExternalExecution pending = external(pendingEvent);
        if (!Objects.equals(run.getWaitingToolName(), pending.toolName())) {
            throw conflict("外部执行工具身份不匹配");
        }
        requireSameExpiry(run, pending.expiresAt());

        long newEpoch = incrementOwnerEpoch(run);
        LocalDateTime leaseUntil = leaseUntil(now, command.ownerLease(), run.getDeadlineAt());
        long sequence = nextSequence(run);
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("internalExecutorId", command.internalExecutorId())
                .put("toolCallId", command.toolCallId())
                .put("toolName", pending.toolName())
                .put("ownerInstanceId", command.newOwnerInstanceId())
                .put("ownerEpoch", newEpoch);
        payload.set("result", parseObject(
                command.resultPayloadJson(), "resultPayloadJson"));
        insertEvent(
                run,
                sequence,
                "EXTERNAL_EXECUTION_RESULT",
                WAITING_SOURCE,
                null,
                command.toolCallId(),
                null,
                payload,
                now);
        if (runMapper.resumeExternal(
                run.getId(),
                run.getOwnerEpoch(),
                command.toolCallId(),
                command.newOwnerInstanceId(),
                newEpoch,
                leaseUntil,
                increment(sequence),
                now) != 1) {
            throw conflict("外部执行请求已由其他节点处理");
        }
        return resumed(run, command.newOwnerInstanceId(), newEpoch, leaseUntil);
    }

    private void requireCheckpoint(AgentRun run, WaitingCheckpoint checkpoint) {
        if (!Objects.equals(run.getAgentStateSessionId(), checkpoint.sessionId())
                || !Objects.equals(run.getKernelFingerprint(), checkpoint.kernelFingerprint())
                || !Objects.equals(
                        run.getAgentDefinitionSnapshotJson(),
                        checkpoint.agentDefinitionSnapshotJson())) {
            throw conflict("WAITING 检查点与持久化运行身份不一致");
        }
        long lastCommitted = nextSequence(run) - 1;
        if (checkpoint.pausedThroughSequence() != lastCommitted) {
            throw conflict("WAITING 检查点事件游标不是最新已提交序号");
        }
    }

    private PendingConfirmation normalize(
            PendingConfirmation candidate, AgentRun run, LocalDateTime now) {
        LocalDateTime expiresAt = clampedExpiry(candidate.expiresAt(), run, now);
        return new PendingConfirmation(
                candidate.replyId(),
                candidate.decisionIds(),
                candidate.pendingToolCallsJson(),
                toInstant(expiresAt));
    }

    private LocalDateTime clampedExpiry(
            Instant requested, AgentRun run, LocalDateTime now) {
        LocalDateTime requestedTime = toDatabaseTime(requested);
        LocalDateTime deadline = Objects.requireNonNull(
                run.getDeadlineAt(), "run.deadlineAt must not be null");
        LocalDateTime expiresAt = requestedTime.isAfter(deadline)
                ? deadline
                : requestedTime;
        if (!expiresAt.isAfter(now)) {
            throw conflict("WAITING 请求已过期");
        }
        return expiresAt;
    }

    private LocalDateTime requireActionableExpiry(
            Instant persistedExpiry, AgentRun run, LocalDateTime now) {
        LocalDateTime expiresAt = toDatabaseTime(persistedExpiry);
        if (expiresAt.isAfter(run.getDeadlineAt()) || !expiresAt.isAfter(now)) {
            throw conflict("WAITING 请求已过期或超过运行截止时间");
        }
        return expiresAt;
    }

    private void requireWaiting(
            AgentRun run, AgentRunStatus expected, LocalDateTime now) {
        if (!expected.name().equals(run.getStatus())) {
            throw conflict("Agent 运行当前不在 " + expected.name() + " 状态");
        }
        if (run.getWaitExpiresAt() == null
                || !run.getWaitExpiresAt().isAfter(now)
                || run.getDeadlineAt() == null
                || !run.getDeadlineAt().isAfter(now)) {
            throw conflict("WAITING 请求已过期");
        }
    }

    private void requireSameExpiry(AgentRun run, Instant candidateExpiry) {
        if (!Objects.equals(run.getWaitExpiresAt(), toDatabaseTime(candidateExpiry))) {
            throw conflict("WAITING 候选过期时间与运行状态不一致");
        }
    }

    private AgentRun requireRunForUpdate(String runId) {
        AgentRun run = runMapper.selectByRunIdForUpdate(runId);
        if (run == null) {
            throw new BusinessException(404, "Agent 运行不存在");
        }
        return run;
    }

    private AgentRun requireAuthorizedRun(String runId, long currentUserId) {
        AgentRun run = requireRunForUpdate(runId);
        if (!Objects.equals(run.getUserId(), currentUserId)) {
            throw new BusinessException(403, "无权访问该 Agent 运行");
        }
        return run;
    }

    private boolean isCurrentRunningOwner(
            AgentRun run, Long expectedOwnerEpoch, LocalDateTime now) {
        return AgentRunStatus.RUNNING.name().equals(run.getStatus())
                && Objects.equals(run.getOwnerEpoch(), expectedOwnerEpoch)
                && run.getOwnerInstanceId() != null
                && !run.getOwnerInstanceId().isBlank()
                && run.getLeaseUntil() != null
                && run.getLeaseUntil().isAfter(now)
                && run.getDeadlineAt() != null
                && run.getDeadlineAt().isAfter(now);
    }

    private PendingConfirmation confirmation(AgentEvent event) {
        ObjectNode payload = parseObject(event.getPayloadJson(), "confirmation payload");
        JsonNode decisions = payload.get("decisionIds");
        if (decisions == null || !decisions.isArray()) {
            throw new IllegalStateException("Persisted confirmation decisionIds are invalid");
        }
        Set<String> decisionIds = new LinkedHashSet<>();
        decisions.forEach(value -> decisionIds.add(requireJsonText(
                value, "confirmation decisionId")));
        return new PendingConfirmation(
                requireJsonText(payload.get("replyId"), "confirmation replyId"),
                decisionIds,
                requireJsonText(
                        payload.get("pendingToolCallsJson"),
                        "confirmation pendingToolCallsJson"),
                parseInstant(payload.get("expiresAt"), "confirmation expiresAt"));
    }

    private PendingExternalExecution external(AgentEvent event) {
        ObjectNode payload = parseObject(event.getPayloadJson(), "external payload");
        return new PendingExternalExecution(
                requireJsonText(payload.get("toolCallId"), "external toolCallId"),
                requireJsonText(payload.get("toolName"), "external toolName"),
                requireJsonText(
                        payload.get("suspendedPayloadJson"),
                        "external suspendedPayloadJson"),
                parseInstant(payload.get("expiresAt"), "external expiresAt"));
    }

    private void validateConfirmation(PendingConfirmation candidate) {
        requireText(candidate.replyId(), 128, "candidate.replyId");
        ArrayNode tools = parseArray(
                candidate.pendingToolCallsJson(), "pendingToolCallsJson");
        if (tools.isEmpty()) {
            throw new IllegalArgumentException("pendingToolCallsJson must not be empty");
        }
        Set<String> toolCallIds = new LinkedHashSet<>();
        for (JsonNode tool : tools) {
            if (!tool.isObject()) {
                throw new IllegalArgumentException(
                        "pendingToolCallsJson items must be objects");
            }
            String toolCallId = requireJsonText(
                    tool.get("toolCallId"), "pending toolCallId");
            requireText(toolCallId, 128, "pending toolCallId");
            requireText(
                    requireJsonText(tool.get("toolName"), "pending toolName"),
                    128,
                    "pending toolName");
            requireJsonText(
                    tool.get("argumentsPreview"), "pending argumentsPreview");
            if (!toolCallIds.add(toolCallId)) {
                throw new IllegalArgumentException(
                        "pendingToolCallsJson contains duplicate toolCallId");
            }
        }
        if (!toolCallIds.equals(candidate.decisionIds())) {
            throw new IllegalArgumentException(
                    "decisionIds must exactly match pending toolCallIds");
        }
    }

    private void validateExternal(PendingExternalExecution pending) {
        requireText(pending.toolCallId(), 128, "pending.toolCallId");
        requireText(pending.toolName(), 128, "pending.toolName");
        parseObject(pending.suspendedPayloadJson(), "suspendedPayloadJson");
    }

    private void insertEvent(
            AgentRun run,
            long sequence,
            String rawEventType,
            String source,
            String replyId,
            String toolCallId,
            String outputType,
            JsonNode payload,
            LocalDateTime now) {
        boolean publishRequired = outputType != null;
        AgentEvent event = AgentEvent.builder()
                .runId(run.getRunId())
                .sequenceNo(sequence)
                .schemaVersion(1)
                .rawEventId(stableRawEventId(
                        rawEventType, run.getRunId(), replyId, toolCallId))
                .rawEventType(rawEventType)
                .source(source)
                .replyId(replyId)
                .toolCallId(toolCallId)
                .parentToolCallId(run.getParentToolCallId())
                .agentName(run.getAgentName())
                .outputType(outputType)
                .payloadJson(writeJson(payload))
                .eventCreatedAt(now)
                .publishRequired(publishRequired)
                .publishStatus(publishRequired ? "PENDING" : "NOT_REQUIRED")
                .nextPublishAttemptAt(publishRequired ? now : null)
                .build();
        if (eventMapper.insert(event) != 1) {
            throw new IllegalStateException(
                    "WAITING event insert did not affect exactly one row");
        }
    }

    private ResumedAgentRun resumed(
            AgentRun run,
            String ownerInstanceId,
            long ownerEpoch,
            LocalDateTime leaseUntil) {
        Long pausedThrough = run.getPausedThroughSequence();
        if (pausedThrough == null || pausedThrough < 0) {
            throw new IllegalStateException(
                    "Persisted WAITING run has no valid paused sequence");
        }
        return new ResumedAgentRun(
                run.getRunId(),
                run.getConversationId(),
                run.getAgentStateSessionId(),
                run.getKernelFingerprint(),
                run.getAgentDefinitionSnapshotJson(),
                pausedThrough,
                ownerInstanceId,
                ownerEpoch,
                toInstant(leaseUntil),
                toInstant(run.getDeadlineAt()));
    }

    private long nextSequence(AgentRun run) {
        Long sequence = run.getNextSequence();
        if (sequence == null || sequence < 1) {
            throw new IllegalStateException("Agent run has an invalid next sequence");
        }
        return sequence;
    }

    private long incrementOwnerEpoch(AgentRun run) {
        Long epoch = run.getOwnerEpoch();
        if (epoch == null || epoch < 1) {
            throw new IllegalStateException("Agent run has an invalid owner epoch");
        }
        return increment(epoch);
    }

    private long increment(long value) {
        try {
            return Math.addExact(value, 1L);
        } catch (ArithmeticException overflow) {
            throw new IllegalStateException("Agent run counter overflow", overflow);
        }
    }

    private LocalDateTime leaseUntil(
            LocalDateTime now,
            java.time.Duration lease,
            LocalDateTime deadline) {
        LocalDateTime requested = now.plus(lease).truncatedTo(ChronoUnit.MILLIS);
        return requested.isAfter(deadline) ? deadline : requested;
    }

    private boolean sameConfirmation(
            PendingConfirmation left, PendingConfirmation right) {
        return Objects.equals(left.replyId(), right.replyId())
                && Objects.equals(left.decisionIds(), right.decisionIds())
                && Objects.equals(left.expiresAt(), right.expiresAt())
                && Objects.equals(
                        parseJson(left.pendingToolCallsJson(), "pendingToolCallsJson"),
                        parseJson(right.pendingToolCallsJson(), "pendingToolCallsJson"));
    }

    private boolean sameExternal(
            PendingExternalExecution left, PendingExternalExecution right) {
        return Objects.equals(left.toolCallId(), right.toolCallId())
                && Objects.equals(left.toolName(), right.toolName())
                && Objects.equals(left.expiresAt(), right.expiresAt())
                && Objects.equals(
                        parseJson(left.suspendedPayloadJson(), "suspendedPayloadJson"),
                        parseJson(right.suspendedPayloadJson(), "suspendedPayloadJson"));
    }

    private ArrayNode textArray(Set<String> values) {
        ArrayNode array = JsonNodeFactory.instance.arrayNode();
        values.stream().sorted(Comparator.naturalOrder()).forEach(array::add);
        return array;
    }

    private JsonNode parseJson(String value, String field) {
        requireJsonSize(value, field);
        try {
            JsonNode parsed = objectMapper.readTree(value);
            if (parsed == null) {
                throw new IllegalArgumentException(field + " must contain JSON");
            }
            return parsed;
        } catch (JsonProcessingException invalidJson) {
            throw new IllegalArgumentException(field + " must contain valid JSON", invalidJson);
        }
    }

    private ObjectNode parseObject(String value, String field) {
        JsonNode parsed = parseJson(value, field);
        if (!parsed.isObject()) {
            throw new IllegalArgumentException(field + " must be a JSON object");
        }
        return (ObjectNode) parsed;
    }

    private ArrayNode parseArray(String value, String field) {
        JsonNode parsed = parseJson(value, field);
        if (!parsed.isArray()) {
            throw new IllegalArgumentException(field + " must be a JSON array");
        }
        return (ArrayNode) parsed;
    }

    private void requireJsonSize(String value, String field) {
        if (value.getBytes(StandardCharsets.UTF_8).length > MAX_JSON_BYTES) {
            throw new IllegalArgumentException(field + " exceeds 8 MiB");
        }
    }

    private String writeJson(JsonNode value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException serializationFailure) {
            throw new IllegalStateException(
                    "WAITING event payload serialization failed", serializationFailure);
        }
    }

    private String stableRawEventId(
            String type, String runId, String replyId, String toolCallId) {
        String material = type + '\0' + runId + '\0'
                + Objects.toString(replyId, "") + '\0'
                + Objects.toString(toolCallId, "");
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(material.getBytes(StandardCharsets.UTF_8));
            return "platform-" + HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private Instant parseInstant(JsonNode value, String field) {
        try {
            return Instant.parse(requireJsonText(value, field));
        } catch (RuntimeException invalid) {
            throw new IllegalStateException(field + " is invalid", invalid);
        }
    }

    private String requireJsonText(JsonNode value, String field) {
        if (value == null || !value.isTextual() || value.textValue().isBlank()) {
            throw new IllegalStateException(field + " must be non-blank text");
        }
        return value.textValue();
    }

    private LocalDateTime toDatabaseTime(Instant value) {
        return LocalDateTime.ofInstant(
                value.truncatedTo(ChronoUnit.MILLIS), ZoneOffset.UTC);
    }

    private Instant toInstant(LocalDateTime value) {
        return Objects.requireNonNull(value, "database time must not be null")
                .toInstant(ZoneOffset.UTC);
    }

    private String requireText(String value, int maxLength, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        if (value.length() > maxLength) {
            throw new IllegalArgumentException(
                    field + " must not exceed " + maxLength + " characters");
        }
        return value;
    }

    private void requirePositive(long value, String field) {
        if (value <= 0) {
            throw new IllegalArgumentException(field + " must be positive");
        }
    }

    private BusinessException conflict(String message) {
        return new BusinessException(409, message);
    }

    private <T> Mono<T> transactional(java.util.function.Supplier<T> operation) {
        return Mono.fromCallable(() -> Objects.requireNonNull(
                        transactions.execute(ignored -> operation.get()),
                        "WAITING transaction returned no result"))
                .subscribeOn(schedulers.journal());
    }
}
