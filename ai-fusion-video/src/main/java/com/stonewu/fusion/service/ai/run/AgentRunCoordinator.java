package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.ChildRunAdmission;
import com.stonewu.fusion.service.ai.run.model.ChildRunIdentityConflictException;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartChildAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import lombok.RequiredArgsConstructor;
import org.springframework.dao.DuplicateKeyException;
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
import java.util.HexFormat;
import java.util.Objects;

/**
 * Owns the only root and platform-child run admission transactions.
 */
@Service
@RequiredArgsConstructor
public class AgentRunCoordinator {

    private static final long INITIAL_OWNER_EPOCH = 1L;

    private final AgentRunRepository runRepository;
    private final AgentMessageAllocator messageAllocator;
    private final TransactionTemplate transactionTemplate;
    private final AgentRuntimeSchedulers schedulers;

    public Mono<StartedAgentRun> start(StartAgentRunCommand command) {
        Objects.requireNonNull(command, "command must not be null");
        return Mono.fromCallable(() -> requireTransactionResult(
                        transactionTemplate.execute(ignored -> startTransaction(command))))
                .subscribeOn(schedulers.journal());
    }

    public Mono<ChildRunAdmission> startChild(StartChildAgentRunCommand command) {
        Objects.requireNonNull(command, "command must not be null");
        return Mono.fromCallable(() -> requireTransactionResult(
                        transactionTemplate.execute(ignored -> startChildTransaction(command))))
                .subscribeOn(schedulers.journal());
    }

    private StartedAgentRun startTransaction(StartAgentRunCommand command) {
        requireRootIdentity(command);
        AgentConversation conversation = requireOwnedConversation(
                command.conversationId(), command.userId(), command.projectId());
        LocalDateTime databaseNow = runRepository.databaseNow();
        LocalDateTime deadline = requireFutureDeadline(command.deadline(), databaseNow);
        LocalDateTime leaseUntil = leaseUntil(databaseNow, command.ownerLease(), deadline);

        AgentRun run = AgentRun.builder()
                .runId(command.runId())
                .conversationId(conversation.getConversationId())
                .userId(conversation.getUserId())
                .projectId(conversation.getProjectId())
                .agentType(command.agentType())
                .kernelFingerprint(command.kernelSnapshot().fingerprint())
                .agentDefinitionSnapshotJson(command.kernelSnapshot().snapshotJson())
                .agentStateSessionId(command.agentStateSessionId())
                .status(AgentRunStatus.RUNNING.name())
                .ownerInstanceId(command.ownerInstanceId())
                .ownerEpoch(INITIAL_OWNER_EPOCH)
                .leaseUntil(leaseUntil)
                .deadlineAt(deadline)
                .startedAt(databaseNow)
                .heartbeatAt(databaseNow)
                .build();
        try {
            runRepository.insert(run);
        } catch (DuplicateKeyException duplicate) {
            throw new BusinessException(409, "该会话已有运行中的 Agent 任务");
        }

        long initialOrder = messageAllocator.append(
                conversation.getConversationId(), userMessage(
                        command.runId(), command.userContent(), command.referencesJson()));
        return started(run, command.kernelSnapshot(), initialOrder);
    }

    private ChildRunAdmission startChildTransaction(StartChildAgentRunCommand command) {
        AgentRun parent = runRepository.lockRun(command.parentRunId());
        LocalDateTime databaseNow = runRepository.databaseNow();
        requireAdmissibleParent(parent, command, databaseNow);

        LocalDateTime deadline = requireFutureDeadline(command.deadline(), databaseNow);
        if (deadline.isAfter(parent.getDeadlineAt())) {
            throw new IllegalArgumentException("Child run deadline must not exceed parent deadline");
        }
        LocalDateTime leaseUntil = leaseUntil(databaseNow, command.ownerLease(), deadline);
        AgentRun child = AgentRun.builder()
                .runId(command.childRunId())
                .conversationId(parent.getConversationId())
                .userId(parent.getUserId())
                .projectId(parent.getProjectId())
                .agentType(command.agentDefinitionStableKey())
                .parentRunId(parent.getRunId())
                .parentToolCallId(command.parentToolCallId())
                .agentName(command.agentName())
                .kernelFingerprint(command.kernelSnapshot().fingerprint())
                .agentDefinitionSnapshotJson(command.kernelSnapshot().snapshotJson())
                .agentStateSessionId(childSessionId(parent, command))
                .status(AgentRunStatus.RUNNING.name())
                .ownerInstanceId(command.ownerInstanceId())
                .ownerEpoch(INITIAL_OWNER_EPOCH)
                .leaseUntil(leaseUntil)
                .deadlineAt(deadline)
                .startedAt(databaseNow)
                .heartbeatAt(databaseNow)
                .build();
        try {
            runRepository.insert(child);
            long initialOrder = messageAllocator.append(
                    parent.getConversationId(), userMessage(
                            child.getRunId(), command.userContent(), command.referencesJson()));
            return new ChildRunAdmission(
                    started(child, command.kernelSnapshot(), initialOrder),
                    AgentRunStatus.RUNNING,
                    true);
        } catch (DuplicateKeyException duplicate) {
            return existingChildAdmission(command, deadline, duplicate);
        }
    }

    private ChildRunAdmission existingChildAdmission(
            StartChildAgentRunCommand command,
            LocalDateTime persistedDeadline,
            DuplicateKeyException duplicate) {
        AgentRun existing = runRepository.lockChild(
                command.parentRunId(), command.parentToolCallId());
        if (existing == null) {
            throw duplicate;
        }
        if (!Objects.equals(existing.getAgentName(), command.agentName())
                || !Objects.equals(existing.getAgentType(), command.agentDefinitionStableKey())
                || !Objects.equals(existing.getKernelFingerprint(),
                        command.kernelSnapshot().fingerprint())
                || !Objects.equals(existing.getAgentDefinitionSnapshotJson(),
                        command.kernelSnapshot().snapshotJson())
                || !Objects.equals(existing.getDeadlineAt(), persistedDeadline)) {
            throw new ChildRunIdentityConflictException(
                    command.parentRunId(), command.parentToolCallId());
        }
        Long initialOrder = runRepository.findInitialMessageOrder(existing.getRunId());
        if (initialOrder == null || initialOrder < 1) {
            throw new IllegalStateException(
                    "Existing child run has no persisted initial message: " + existing.getRunId());
        }
        AgentRunStatus status;
        try {
            status = AgentRunStatus.valueOf(existing.getStatus());
        } catch (IllegalArgumentException invalidStatus) {
            throw new IllegalStateException(
                    "Existing child run has an unsupported status: " + existing.getStatus(),
                    invalidStatus);
        }
        return new ChildRunAdmission(
                started(existing, command.kernelSnapshot(), initialOrder), status, false);
    }

    private AgentConversation requireOwnedConversation(
            String conversationId, long userId, Long requestedProjectId) {
        AgentConversation conversation = runRepository.lockConversation(conversationId);
        if (conversation == null) {
            throw new BusinessException(404, "Agent 对话不存在");
        }
        if (!Objects.equals(conversation.getUserId(), userId)) {
            throw new BusinessException(403, "无权在该 Agent 对话中启动任务");
        }
        if (requestedProjectId != null
                && !Objects.equals(conversation.getProjectId(), requestedProjectId)) {
            throw new BusinessException(409, "Agent 对话与请求项目不一致");
        }
        return conversation;
    }

    private void requireRootIdentity(StartAgentRunCommand command) {
        if (command.parentRunId() != null
                || command.parentToolCallId() != null
                || command.agentName() != null) {
            throw new IllegalArgumentException(
                    "Root run admission does not accept parent identity fields");
        }
    }

    private void requireAdmissibleParent(
            AgentRun parent,
            StartChildAgentRunCommand command,
            LocalDateTime databaseNow) {
        if (parent == null) {
            throw new BusinessException(404, "父 Agent 运行不存在");
        }
        if (!AgentRunStatus.RUNNING.name().equals(parent.getStatus())) {
            throw new BusinessException(409, "父 Agent 运行当前不接受子任务");
        }
        if (!Objects.equals(parent.getOwnerInstanceId(), command.parentOwnerInstanceId())
                || !Objects.equals(parent.getOwnerEpoch(), command.parentOwnerEpoch())) {
            throw new BusinessException(409, "父 Agent 运行所有权已变更");
        }
        if (parent.getLeaseUntil() == null || !parent.getLeaseUntil().isAfter(databaseNow)) {
            throw new BusinessException(409, "父 Agent 运行租约已失效");
        }
        if (parent.getDeadlineAt() == null || !parent.getDeadlineAt().isAfter(databaseNow)) {
            throw new BusinessException(409, "父 Agent 运行已超过截止时间");
        }
    }

    private LocalDateTime requireFutureDeadline(Instant deadline, LocalDateTime databaseNow) {
        LocalDateTime persisted = toDatabaseTime(deadline);
        if (!persisted.isAfter(databaseNow)) {
            throw new IllegalArgumentException("Agent run deadline must be in the future");
        }
        return persisted;
    }

    private LocalDateTime leaseUntil(
            LocalDateTime databaseNow,
            java.time.Duration ownerLease,
            LocalDateTime deadline) {
        LocalDateTime requested = databaseNow.plus(ownerLease)
                .truncatedTo(ChronoUnit.MILLIS);
        return requested.isAfter(deadline) ? deadline : requested;
    }

    private AgentMessage userMessage(String runId, String content, String referencesJson) {
        return AgentMessage.builder()
                .runId(runId)
                .role("user")
                .content(content)
                .referencesJson(referencesJson)
                .build();
    }

    private StartedAgentRun started(
            AgentRun run,
            com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot snapshot,
            long initialMessageOrder) {
        return new StartedAgentRun(
                run.getRunId(),
                run.getConversationId(),
                run.getAgentStateSessionId(),
                run.getOwnerInstanceId(),
                run.getOwnerEpoch(),
                toInstant(run.getLeaseUntil()),
                toInstant(run.getDeadlineAt()),
                snapshot,
                initialMessageOrder);
    }

    private String childSessionId(AgentRun parent, StartChildAgentRunCommand command) {
        String material = parent.getRunId() + '\0'
                + parent.getAgentStateSessionId() + '\0'
                + command.parentToolCallId() + '\0'
                + command.agentDefinitionStableKey();
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(material.getBytes(StandardCharsets.UTF_8));
            return "afv-child:" + HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private LocalDateTime toDatabaseTime(Instant instant) {
        return LocalDateTime.ofInstant(
                instant.truncatedTo(ChronoUnit.MILLIS), ZoneOffset.UTC);
    }

    private Instant toInstant(LocalDateTime value) {
        return value.toInstant(ZoneOffset.UTC);
    }

    private <T> T requireTransactionResult(T value) {
        return Objects.requireNonNull(value, "Agent run transaction returned no result");
    }
}
