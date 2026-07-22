package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.AgentWaitingStatePort;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.PendingConfirmation;
import com.stonewu.fusion.service.ai.run.model.PendingExternalExecution;
import com.stonewu.fusion.service.ai.run.model.ResumeConfirmationCommand;
import com.stonewu.fusion.service.ai.run.model.ResumeExternalCommand;
import com.stonewu.fusion.service.ai.run.model.ResumedAgentRun;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import com.stonewu.fusion.service.ai.run.model.WaitingCheckpoint;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@Testcontainers(disabledWithoutDocker = false)
class AgentWaitingStateIT {

    private static final long USER_ID = 42L;
    private static final long PROJECT_ID = 77L;
    private static final String PENDING_TOOLS = """
            [{"toolCallId":"tool-a","toolName":"generate_image",\
            "argumentsPreview":"{\\"prompt\\":\\"sunrise\\"}"}]
            """;

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
                "fusion.agentscope.v2.execution.instance-id", () -> "waiting-node-a");
    }

    @Autowired
    private AgentWaitingStatePort waiting;

    @Autowired
    private AgentRunCoordinator coordinator;

    @Autowired
    private AgentConversationService conversationService;

    @Autowired
    private AgentRunMapper runMapper;

    @Autowired
    private AgentEventMapper eventMapper;

    @Test
    void confirmationIsInvisibleUntilCheckpointAndResumesOnceAcrossNodes()
            throws Exception {
        StartedAgentRun started = startRun(Duration.ofMinutes(2));
        PendingConfirmation candidate = confirmation(
                started.deadline().minusSeconds(30));

        await(waiting.recordConfirmationCandidate(started.runId(), candidate));
        await(waiting.recordConfirmationCandidate(started.runId(), candidate));
        assertThat(events(started.runId(), "REQUIRE_USER_CONFIRM")).hasSize(1);
        AgentEvent rawCandidate = events(
                started.runId(), "REQUIRE_USER_CONFIRM").getFirst();
        assertThat(rawCandidate.getOutputType()).isNull();
        assertThat(rawCandidate.getPublishRequired()).isFalse();
        assertConflict(() -> await(waiting.getPendingConfirmationAuthorized(
                started.runId(), USER_ID, candidate.replyId())));

        AgentRun beforeWaiting = run(started.runId());
        WaitingCheckpoint checkpoint = checkpoint(beforeWaiting);
        assertThat(await(waiting.enterWaitingConfirmation(
                started.runId(), started.ownerEpoch() + 1, checkpoint))).isFalse();
        assertThat(await(waiting.enterWaitingConfirmation(
                started.runId(), started.ownerEpoch(), checkpoint))).isTrue();

        AgentRun persistedWaiting = run(started.runId());
        assertThat(persistedWaiting.getStatus())
                .isEqualTo(AgentRunStatus.WAITING_CONFIRMATION.name());
        assertThat(persistedWaiting.getOwnerInstanceId()).isNull();
        assertThat(persistedWaiting.getLeaseUntil()).isNull();
        assertThat(persistedWaiting.getOwnerEpoch()).isEqualTo(started.ownerEpoch());
        assertThat(persistedWaiting.getPausedThroughSequence())
                .isEqualTo(checkpoint.pausedThroughSequence());
        AgentEvent actionable = events(
                started.runId(), "PLATFORM_USER_CONFIRM_REQUIRED").getFirst();
        assertThat(actionable.getOutputType()).isEqualTo("TOOL_CALL");
        assertThat(actionable.getPublishRequired()).isTrue();

        assertForbidden(() -> await(waiting.getPendingConfirmationAuthorized(
                started.runId(), USER_ID + 1, candidate.replyId())));
        assertThat(await(waiting.getPendingConfirmationAuthorized(
                started.runId(), USER_ID, candidate.replyId())))
                .isEqualTo(candidate);
        assertConflict(() -> await(waiting.resumeConfirmation(
                new ResumeConfirmationCommand(
                        started.runId(),
                        USER_ID,
                        candidate.replyId(),
                        Set.of("different-tool"),
                        "waiting-node-b",
                        Duration.ofSeconds(30)))));

        CountDownLatch start = new CountDownLatch(1);
        Object left;
        Object right;
        try (ExecutorService pool = Executors.newFixedThreadPool(2)) {
            Future<Object> first = pool.submit(() -> resumeConfirmation(
                    start, started.runId(), candidate, "waiting-node-b"));
            Future<Object> second = pool.submit(() -> resumeConfirmation(
                    start, started.runId(), candidate, "waiting-node-c"));
            start.countDown();
            left = first.get(15, TimeUnit.SECONDS);
            right = second.get(15, TimeUnit.SECONDS);
        }
        List<Object> outcomes = List.of(left, right);
        assertThat(outcomes.stream().filter(ResumedAgentRun.class::isInstance)).hasSize(1);
        assertThat(outcomes.stream().filter(BusinessException.class::isInstance)).hasSize(1);
        ResumedAgentRun resumed = outcomes.stream()
                .filter(ResumedAgentRun.class::isInstance)
                .map(ResumedAgentRun.class::cast)
                .findFirst()
                .orElseThrow();
        assertResumedIdentity(started, checkpoint, resumed);
        assertThat(resumed.newOwnerEpoch()).isEqualTo(started.ownerEpoch() + 1);
        assertThat(resumed.newOwnerInstanceId())
                .isIn("waiting-node-b", "waiting-node-c");

        AgentRun running = run(started.runId());
        assertThat(running.getStatus()).isEqualTo(AgentRunStatus.RUNNING.name());
        assertThat(running.getOwnerInstanceId())
                .isEqualTo(resumed.newOwnerInstanceId());
        assertThat(running.getWaitingReplyId()).isNull();
        assertThat(running.getWaitExpiresAt()).isNull();
        assertThat(events(started.runId(), "USER_CONFIRM_RESULT")).hasSize(1);
    }

    @Test
    void externalExecutionClampsDeadlineAndRestoresExactPersistedIdentity() {
        StartedAgentRun started = startRun(Duration.ofSeconds(75));
        AgentRun running = run(started.runId());
        WaitingCheckpoint checkpoint = checkpoint(running);
        PendingExternalExecution requested = new PendingExternalExecution(
                "external-tool-1",
                "asset_renderer",
                "{\"resumeToken\":\"opaque-1\"}",
                started.deadline().plusSeconds(60));

        assertThat(await(waiting.enterWaitingExternal(
                started.runId(), started.ownerEpoch(), checkpoint, requested))).isTrue();
        AgentRun persistedWaiting = run(started.runId());
        assertThat(persistedWaiting.getStatus())
                .isEqualTo(AgentRunStatus.WAITING_EXTERNAL.name());
        assertThat(persistedWaiting.getWaitExpiresAt().toInstant(ZoneOffset.UTC))
                .isEqualTo(started.deadline());
        assertConflict(() -> await(waiting.getPendingExternalAuthorized(
                started.runId(), USER_ID, "wrong-tool")));
        PendingExternalExecution pending = await(waiting.getPendingExternalAuthorized(
                started.runId(), USER_ID, requested.toolCallId()));
        assertThat(pending.toolName()).isEqualTo(requested.toolName());
        assertThat(pending.suspendedPayloadJson())
                .isEqualTo(requested.suspendedPayloadJson());
        assertThat(pending.expiresAt()).isEqualTo(started.deadline());

        ResumedAgentRun resumed = await(waiting.resumeExternal(
                new ResumeExternalCommand(
                        started.runId(),
                        USER_ID,
                        "executor-production-1",
                        requested.toolCallId(),
                        "{\"status\":\"SUCCEEDED\",\"assetId\":99}",
                        "waiting-node-z",
                        Duration.ofSeconds(30))));
        assertResumedIdentity(started, checkpoint, resumed);
        assertThat(resumed.newOwnerInstanceId()).isEqualTo("waiting-node-z");
        assertThat(resumed.newOwnerEpoch()).isEqualTo(started.ownerEpoch() + 1);
        AgentEvent result = events(
                started.runId(), "EXTERNAL_EXECUTION_RESULT").getFirst();
        assertThat(result.getToolCallId()).isEqualTo(requested.toolCallId());
        assertThat(result.getOutputType()).isNull();
        assertThat(result.getPayloadJson()).contains("executor-production-1");
    }

    @Test
    void rejectsCheckpointDriftExpiredWaitAndConflictingCandidate() {
        StartedAgentRun started = startRun(Duration.ofMinutes(2));
        PendingConfirmation candidate = confirmation(
                started.deadline().minusSeconds(30));
        await(waiting.recordConfirmationCandidate(started.runId(), candidate));

        PendingConfirmation conflict = new PendingConfirmation(
                candidate.replyId(),
                candidate.decisionIds(),
                "[{\"toolCallId\":\"tool-a\",\"toolName\":\"other\","
                        + "\"argumentsPreview\":\"{}\"}]",
                candidate.expiresAt());
        assertConflict(() -> await(waiting.recordConfirmationCandidate(
                started.runId(), conflict)));
        AgentRun running = run(started.runId());
        WaitingCheckpoint wrongSession = new WaitingCheckpoint(
                running.getAgentStateSessionId() + "-tampered",
                running.getKernelFingerprint(),
                running.getAgentDefinitionSnapshotJson(),
                running.getNextSequence() - 1);
        assertConflict(() -> await(waiting.enterWaitingConfirmation(
                started.runId(), started.ownerEpoch(), wrongSession)));
        assertThat(run(started.runId()).getStatus())
                .isEqualTo(AgentRunStatus.RUNNING.name());

        WaitingCheckpoint checkpoint = checkpoint(run(started.runId()));
        assertThat(await(waiting.enterWaitingConfirmation(
                started.runId(), started.ownerEpoch(), checkpoint))).isTrue();
        LocalDateTime expired = LocalDateTime.now(ZoneOffset.UTC).minusSeconds(1);
        assertThat(runMapper.update(
                null,
                new LambdaUpdateWrapper<AgentRun>()
                        .eq(AgentRun::getRunId, started.runId())
                        .set(AgentRun::getWaitExpiresAt, expired)))
                .isEqualTo(1);
        assertConflict(() -> await(waiting.getPendingConfirmationAuthorized(
                started.runId(), USER_ID, candidate.replyId())));
        assertConflict(() -> await(waiting.resumeConfirmation(
                new ResumeConfirmationCommand(
                        started.runId(),
                        USER_ID,
                        candidate.replyId(),
                        candidate.decisionIds(),
                        "waiting-node-b",
                        Duration.ofSeconds(30)))));
        assertThat(run(started.runId()).getStatus())
                .isEqualTo(AgentRunStatus.WAITING_CONFIRMATION.name());
    }

    private Object resumeConfirmation(
            CountDownLatch start,
            String runId,
            PendingConfirmation candidate,
            String owner) {
        await(start);
        try {
            return await(waiting.resumeConfirmation(new ResumeConfirmationCommand(
                    runId,
                    USER_ID,
                    candidate.replyId(),
                    candidate.decisionIds(),
                    owner,
                    Duration.ofSeconds(30))));
        } catch (BusinessException expectedLoser) {
            return expectedLoser;
        }
    }

    private StartedAgentRun startRun(Duration deadlineAfter) {
        String conversationId = unique("conversation");
        conversationService.createOrUpdate(
                conversationId,
                USER_ID,
                PROJECT_ID,
                "project",
                PROJECT_ID,
                "assistant",
                "waiting test",
                "chat");
        AgentKernelSnapshot snapshot = snapshot();
        Instant deadline = Instant.now().plus(deadlineAfter)
                .truncatedTo(ChronoUnit.MILLIS);
        return await(coordinator.start(new StartAgentRunCommand(
                unique("run"),
                conversationId,
                USER_ID,
                PROJECT_ID,
                "assistant",
                null,
                null,
                null,
                "waiting-session-" + unique("session"),
                snapshot,
                "waiting-node-a",
                Duration.ofSeconds(30),
                deadline,
                "wait for a tool",
                null)));
    }

    private AgentKernelSnapshot snapshot() {
        return new CanonicalAgentKernelSnapshotBuilder().build(
                new AgentKernelSnapshotPayload(
                        AgentKernelSnapshotPayload.CURRENT_SCHEMA_VERSION,
                        "assistant",
                        "assistant",
                        "root",
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

    private PendingConfirmation confirmation(Instant expiresAt) {
        return new PendingConfirmation(
                "reply-1", Set.of("tool-a"), PENDING_TOOLS, expiresAt);
    }

    private WaitingCheckpoint checkpoint(AgentRun run) {
        return new WaitingCheckpoint(
                run.getAgentStateSessionId(),
                run.getKernelFingerprint(),
                run.getAgentDefinitionSnapshotJson(),
                run.getNextSequence() - 1);
    }

    private void assertResumedIdentity(
            StartedAgentRun started,
            WaitingCheckpoint checkpoint,
            ResumedAgentRun resumed) {
        assertThat(resumed.runId()).isEqualTo(started.runId());
        assertThat(resumed.sessionId()).isEqualTo(checkpoint.sessionId());
        assertThat(resumed.kernelFingerprint())
                .isEqualTo(checkpoint.kernelFingerprint());
        assertThat(resumed.agentDefinitionSnapshotJson())
                .isEqualTo(checkpoint.agentDefinitionSnapshotJson());
        assertThat(resumed.pausedThroughSequence())
                .isEqualTo(checkpoint.pausedThroughSequence());
        assertThat(resumed.deadline()).isEqualTo(started.deadline());
    }

    private AgentRun run(String runId) {
        return runMapper.selectOne(new LambdaQueryWrapper<AgentRun>()
                .eq(AgentRun::getRunId, runId));
    }

    private List<AgentEvent> events(String runId, String rawType) {
        return eventMapper.selectList(new LambdaQueryWrapper<AgentEvent>()
                .eq(AgentEvent::getRunId, runId)
                .eq(AgentEvent::getRawEventType, rawType)
                .orderByAsc(AgentEvent::getSequenceNo));
    }

    private void assertConflict(Runnable operation) {
        assertThatThrownBy(operation::run)
                .isInstanceOf(BusinessException.class)
                .satisfies(failure -> assertThat(((BusinessException) failure).getCode())
                        .isEqualTo(409));
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

    private void await(CountDownLatch latch) {
        try {
            if (!latch.await(10, TimeUnit.SECONDS)) {
                throw new IllegalStateException("start latch timed out");
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("start latch interrupted", interrupted);
        }
    }

    private static String unique(String prefix) {
        return prefix + '-' + UUID.randomUUID().toString().replace("-", "");
    }
}
