package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentRuntimeErrorCode;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.run.AgentExecution;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.AgentRunReconciliationService;
import com.stonewu.fusion.service.ai.run.AgentRunRedisSignalService;
import com.stonewu.fusion.service.ai.run.AgentWaitingStatePort;
import com.stonewu.fusion.service.ai.run.CancellationCoordinator;
import com.stonewu.fusion.service.ai.run.OwnedExecutionRegistry;
import com.stonewu.fusion.service.ai.run.RunLeaseGuard;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.ChildRunAdmission;
import com.stonewu.fusion.service.ai.run.model.ExecutionStopReason;
import com.stonewu.fusion.service.ai.run.model.PendingExternalExecution;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartChildAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import com.stonewu.fusion.service.ai.run.model.WaitingCheckpoint;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.transaction.support.TransactionTemplate;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@Testcontainers(disabledWithoutDocker = false)
class AgentFencingCancellationIT {

    private static final long USER_ID = 42L;
    private static final long PROJECT_ID = 77L;

    @Container
    private static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video")
            .withUsername("afv")
            .withPassword("afv-test");

    @DynamicPropertySource
    static void configureDatabase(DynamicPropertyRegistry properties) {
        properties.add("spring.datasource.url", MYSQL::getJdbcUrl);
        properties.add("spring.datasource.username", MYSQL::getUsername);
        properties.add("spring.datasource.password", MYSQL::getPassword);
        properties.add("fusion.agentscope.v2.state.mode", () -> "IN_MEMORY");
        properties.add(
                "fusion.agentscope.v2.execution.instance-id", () -> "fencing-node-a");
        properties.add(
                "fusion.agentscope.v2.execution.maintenance-initial-delay-ms",
                () -> "3600000");
        properties.add(
                "fusion.agentscope.v2.execution.maintenance-delay-ms",
                () -> "3600000");
    }

    @Autowired
    private RunLeaseGuard leases;

    @Autowired
    private CancellationCoordinator cancellations;

    @Autowired
    private AgentRunReconciliationService reconciliation;

    @Autowired
    private AgentRunCoordinator coordinator;

    @Autowired
    private AgentWaitingStatePort waiting;

    @Autowired
    private AgentConversationService conversationService;

    @Autowired
    private AgentRunRepository runRepository;

    @Autowired
    private AgentRunMapper runMapper;

    @Autowired
    private AgentEventMapper eventMapper;

    @Autowired
    private OwnedExecutionRegistry executions;

    @Autowired
    private TransactionTemplate transactions;

    @MockitoBean
    private AgentRunRedisSignalService signals;

    @BeforeEach
    void configureSignals() {
        when(signals.publishCancel(anyString())).thenReturn(Mono.empty());
        when(signals.publishWakeup(anyString(), anyLong())).thenReturn(Mono.empty());
        when(signals.cancellations(anyString())).thenReturn(Flux.never());
    }

    @Test
    void staleEpochIsRejectedAndDatabaseHeartbeatObservesLostCancelSignal() {
        StartedAgentRun started = startRoot("lease", "node-a", Duration.ofMinutes(2));
        await(leases.assertLease(started.runId(), "node-a", started.ownerEpoch()));
        assertThatThrownBy(() -> await(leases.assertLease(
                started.runId(), "node-a", started.ownerEpoch() + 1)))
                .isInstanceOf(RunLeaseGuard.OwnerFencedException.class);
        LocalDateTime leaseBeforeHeartbeat = run(started.runId()).getLeaseUntil();
        await(leases.heartbeat(
                started.runId(),
                started.ownerInstanceId(),
                started.ownerEpoch(),
                Duration.ofSeconds(30)));
        assertThat(run(started.runId()).getLeaseUntil())
                .isAfterOrEqualTo(leaseBeforeHeartbeat);

        AtomicReference<ExecutionStopReason> interrupted = new AtomicReference<>();
        AgentExecution execution = new AgentExecution(
                started.runId(),
                started.ownerInstanceId(),
                started.ownerEpoch(),
                USER_ID,
                started.agentStateSessionId(),
                Flux.never(),
                reason -> Mono.fromRunnable(() -> interrupted.set(reason)),
                () -> { });
        await(executions.registerAndLaunch(
                execution,
                started.deadline(),
                events -> events,
                Flux::then,
                ignored -> Mono.empty()));

        transactions.executeWithoutResult(ignored ->
                runRepository.requestAuthorizedCancellationTree(
                        started.runId(), USER_ID));
        await(leases.heartbeat(
                started.runId(),
                started.ownerInstanceId(),
                started.ownerEpoch(),
                Duration.ofSeconds(30)));

        assertThat(interrupted.get()).isEqualTo(ExecutionStopReason.CANCEL_REQUESTED);
        await(executions.awaitEmpty(Duration.ofSeconds(2)));
        AgentRun cancelled = run(started.runId());
        assertThat(cancelled.getStatus())
                .isEqualTo(AgentRunStatus.CANCEL_REQUESTED.name());
        assertThat(cancelled.getCancelAcknowledgedAt()).isNotNull();
        assertThat(cancelled.getLeaseUntil())
                .isBeforeOrEqualTo(runMapper.selectDatabaseNow());
    }

    @Test
    void cancellationCascadesTheLockedTreeAndLeavesUnrelatedRunsUntouched() {
        StartedAgentRun root = startRoot("tree", "node-root", Duration.ofMinutes(2));
        StartedAgentRun child = startChild(root, "tool-child", "node-child");
        StartedAgentRun grandchild = startChild(
                child, "tool-grandchild", "node-grandchild");
        StartedAgentRun unrelated = startRoot(
                "unrelated", "node-unrelated", Duration.ofMinutes(2));

        assertForbidden(() -> await(cancellations.cancel(
                root.runId(), USER_ID + 1)));
        assertThat(await(cancellations.cancel(root.runId(), USER_ID)))
                .isEqualTo(AgentRunStatus.CANCEL_REQUESTED);

        assertThat(run(root.runId()).getStatus())
                .isEqualTo(AgentRunStatus.CANCEL_REQUESTED.name());
        assertThat(run(child.runId()).getStatus())
                .isEqualTo(AgentRunStatus.CANCEL_REQUESTED.name());
        assertThat(run(grandchild.runId()).getStatus())
                .isEqualTo(AgentRunStatus.CANCEL_REQUESTED.name());
        assertThat(run(unrelated.runId()).getStatus())
                .isEqualTo(AgentRunStatus.RUNNING.name());
    }

    @Test
    void reconciliationRejectsLiveLeaseThenTerminalizesExpiredOwnersOnce() {
        StartedAgentRun ownerLost = startRoot(
                "owner-lost", "node-expiring", Duration.ofMinutes(2));
        await(reconciliation.reconcileExpiredOwner(ownerLost.runId()));
        assertThat(run(ownerLost.runId()).getStatus())
                .isEqualTo(AgentRunStatus.RUNNING.name());

        expireLease(ownerLost.runId());
        await(reconciliation.reconcileExpiredOwner(ownerLost.runId()));
        await(reconciliation.reconcileExpiredOwner(ownerLost.runId()));
        AgentRun failed = run(ownerLost.runId());
        assertThat(failed.getStatus()).isEqualTo(AgentRunStatus.FAILED.name());
        assertThat(failed.getErrorCode())
                .isEqualTo(AgentRuntimeErrorCode.OWNER_LOST.name());
        assertThat(terminalEvents(ownerLost.runId())).hasSize(1);

        StartedAgentRun cancelExpired = startRoot(
                "cancel-expired", "node-cancel", Duration.ofMinutes(2));
        transactions.executeWithoutResult(ignored ->
                runRepository.requestAuthorizedCancellationTree(
                        cancelExpired.runId(), USER_ID));
        expireLease(cancelExpired.runId());
        await(reconciliation.reconcileExpiredOwner(cancelExpired.runId()));
        AgentRun cancelled = run(cancelExpired.runId());
        assertThat(cancelled.getStatus())
                .isEqualTo(AgentRunStatus.CANCELLED.name());
        assertThat(cancelled.getErrorCode())
                .isEqualTo(AgentRuntimeErrorCode.RUN_CANCELLED.name());
        assertThat(terminalEvents(cancelExpired.runId())).hasSize(1);
    }

    @Test
    void ownerlessWaitingRunIsCancelledWithoutWaitingForADeadLease() {
        StartedAgentRun started = startRoot(
                "waiting", "node-waiting", Duration.ofMinutes(2));
        AgentRun running = run(started.runId());
        WaitingCheckpoint checkpoint = new WaitingCheckpoint(
                running.getAgentStateSessionId(),
                running.getKernelFingerprint(),
                running.getAgentDefinitionSnapshotJson(),
                running.getNextSequence() - 1);
        assertThat(await(waiting.enterWaitingExternal(
                started.runId(),
                started.ownerEpoch(),
                checkpoint,
                new PendingExternalExecution(
                        "external-1",
                        "renderer",
                        "{\"resumeToken\":\"one\"}",
                        started.deadline().minusSeconds(10)))))
                .isTrue();
        assertThat(run(started.runId()).getOwnerInstanceId()).isNull();

        assertThat(await(cancellations.cancel(started.runId(), USER_ID)))
                .isEqualTo(AgentRunStatus.CANCELLED);
        assertThat(run(started.runId()).getStatus())
                .isEqualTo(AgentRunStatus.CANCELLED.name());
        assertThat(terminalEvents(started.runId())).hasSize(1);
    }

    private StartedAgentRun startRoot(
            String prefix, String owner, Duration deadlineAfter) {
        String conversationId = unique("conversation-" + prefix);
        conversationService.createOrUpdate(
                conversationId,
                USER_ID,
                PROJECT_ID,
                "project",
                PROJECT_ID,
                "assistant",
                "fencing test",
                "chat");
        Instant deadline = Instant.now().plus(deadlineAfter)
                .truncatedTo(ChronoUnit.MILLIS);
        return await(coordinator.start(new StartAgentRunCommand(
                unique("run-" + prefix),
                conversationId,
                USER_ID,
                PROJECT_ID,
                "assistant",
                null,
                null,
                null,
                unique("session-" + prefix),
                snapshot("assistant"),
                owner,
                Duration.ofSeconds(30),
                deadline,
                "run",
                null)));
    }

    private StartedAgentRun startChild(
            StartedAgentRun parent, String toolCallId, String owner) {
        ChildRunAdmission admission = await(coordinator.startChild(
                new StartChildAgentRunCommand(
                        unique("child"),
                        parent.runId(),
                        toolCallId,
                        parent.ownerInstanceId(),
                        parent.ownerEpoch(),
                        "worker",
                        "worker",
                        snapshot("worker"),
                        owner,
                        Duration.ofSeconds(30),
                        parent.deadline().minusSeconds(1),
                        "child",
                        null)));
        return admission.run();
    }

    private AgentKernelSnapshot snapshot(String stableKey) {
        return new CanonicalAgentKernelSnapshotBuilder().build(
                new AgentKernelSnapshotPayload(
                        AgentKernelSnapshotPayload.CURRENT_SCHEMA_VERSION,
                        stableKey,
                        stableKey,
                        "test",
                        "system",
                        5,
                        "1",
                        1,
                        "openai",
                        "test-model",
                        JsonNodeFactory.instance.objectNode(),
                        List.of(),
                        "test"));
    }

    private void expireLease(String runId) {
        LocalDateTime expired = runMapper.selectDatabaseNow().minusSeconds(1);
        assertThat(runMapper.update(
                null,
                new LambdaUpdateWrapper<AgentRun>()
                        .eq(AgentRun::getRunId, runId)
                        .set(AgentRun::getLeaseUntil, expired)))
                .isEqualTo(1);
    }

    private AgentRun run(String runId) {
        return runMapper.selectOne(new LambdaQueryWrapper<AgentRun>()
                .eq(AgentRun::getRunId, runId));
    }

    private List<AgentEvent> terminalEvents(String runId) {
        return eventMapper.selectList(new LambdaQueryWrapper<AgentEvent>()
                .eq(AgentEvent::getRunId, runId)
                .eq(AgentEvent::getRawEventType, "RUN_TERMINAL"));
    }

    private void assertForbidden(Runnable operation) {
        assertThatThrownBy(operation::run)
                .isInstanceOf(BusinessException.class)
                .satisfies(failure -> assertThat(((BusinessException) failure).getCode())
                        .isEqualTo(403));
    }

    private <T> T await(Mono<T> publisher) {
        return publisher.block(Duration.ofSeconds(15));
    }

    private static String unique(String prefix) {
        return prefix + '-' + UUID.randomUUID().toString().replace("-", "");
    }
}
