package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.AgentStateSessionIds;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.kernel.ToolManifestSnapshot;
import com.stonewu.fusion.service.ai.run.model.ChildRunAdmission;
import com.stonewu.fusion.service.ai.run.model.ChildRunIdentityConflictException;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartChildAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.transaction.support.TransactionTemplate;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.time.Duration;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@Testcontainers(disabledWithoutDocker = false)
class AgentRunStartIT {

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
    }

    @Autowired
    private AgentRunCoordinator coordinator;

    @Autowired
    private AgentConversationService conversationService;

    @Autowired
    private AgentConversationMapper conversationMapper;

    @Autowired
    private AgentMessageMapper messageMapper;

    @Autowired
    private AgentRunMapper runMapper;

    @Autowired
    private TransactionTemplate transactionTemplate;

    @Test
    void startsRootAndMessageAtomicallyAndRejectsInvalidRootAdmissions() {
        long userId = 42L;
        long projectId = 7L;
        String conversationId = createConversation(userId, projectId);
        AgentKernelSnapshot snapshot = snapshot("assistant", 1L);
        Instant deadline = Instant.now().plus(Duration.ofMinutes(10));
        StartAgentRunCommand command = rootCommand(
                uniqueId("root"), conversationId, userId, projectId,
                snapshot, deadline, null, null, null);

        StartedAgentRun started = await(coordinator.start(command));

        assertThat(started.initialMessageOrder()).isEqualTo(1L);
        assertThat(started.deadline()).isEqualTo(deadline.truncatedTo(ChronoUnit.MILLIS));
        AgentRun persisted = run(command.runId());
        assertThat(persisted.getParentRunId()).isNull();
        assertThat(persisted.getParentToolCallId()).isNull();
        assertThat(persisted.getAgentName()).isNull();
        assertThat(persisted.getDeadlineAt()).isNotNull();
        assertThat(messages(command.runId()))
                .singleElement()
                .satisfies(message -> {
                    assertThat(message.getRole()).isEqualTo("user");
                    assertThat(message.getMessageOrder()).isEqualTo(1L);
                    assertThat(message.getContent()).isEqualTo("hello");
                });
        assertConversationCounters(conversationId, 1, 2L);

        StartAgentRunCommand secondRoot = rootCommand(
                uniqueId("second-root"), conversationId, userId, projectId,
                snapshot, deadline, null, null, null);
        assertThatThrownBy(() -> await(coordinator.start(secondRoot)))
                .isInstanceOf(BusinessException.class)
                .satisfies(failure -> assertThat(((BusinessException) failure).getCode())
                        .isEqualTo(409));
        assertConversationCounters(conversationId, 1, 2L);
        assertThat(run(secondRoot.runId())).isNull();

        String pastConversation = createConversation(userId, projectId);
        StartAgentRunCommand pastDeadline = rootCommand(
                uniqueId("past-root"), pastConversation, userId, projectId,
                snapshot, Instant.now().minusSeconds(1), null, null, null);
        assertThatThrownBy(() -> await(coordinator.start(pastDeadline)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("deadline");
        assertThat(run(pastDeadline.runId())).isNull();

        StartAgentRunCommand invalidRootIdentity = rootCommand(
                uniqueId("invalid-root"), pastConversation, userId, projectId,
                snapshot, deadline, "parent", "tool", "child-agent");
        assertThatThrownBy(() -> await(coordinator.start(invalidRootIdentity)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("parent identity");

        String foreignConversation = createConversation(userId, projectId);
        StartAgentRunCommand foreignUser = rootCommand(
                uniqueId("foreign-root"), foreignConversation, 99L, projectId,
                snapshot, deadline, null, null, null);
        assertThatThrownBy(() -> await(coordinator.start(foreignUser)))
                .isInstanceOf(BusinessException.class)
                .satisfies(failure -> assertThat(((BusinessException) failure).getCode())
                        .isEqualTo(403));
        assertThat(run(foreignUser.runId())).isNull();
    }

    @Test
    void rollsBackRunWhenInitialMessageCannotBeInserted() {
        String conversationId = createConversation(42L, 7L);
        AgentMessage conflictingOrder = AgentMessage.builder()
                .conversationId(conversationId)
                .role("user")
                .content("poison-order")
                .messageOrder(1L)
                .build();
        assertThat(messageMapper.insert(conflictingOrder)).isEqualTo(1);

        StartAgentRunCommand command = rootCommand(
                uniqueId("rollback-root"), conversationId, 42L, 7L,
                snapshot("assistant", 1L), Instant.now().plusSeconds(300),
                null, null, null);

        assertThatThrownBy(() -> await(coordinator.start(command)))
                .isInstanceOf(RuntimeException.class);
        assertThat(run(command.runId())).isNull();
        assertConversationCounters(conversationId, 0, 1L);
        assertThat(messages(command.runId())).isEmpty();
    }

    @Test
    void rotatesStateGenerationAfterInterruptedRootAndReusesItAfterSuccess() {
        String conversationId = createConversation(42L, 7L);
        AgentKernelSnapshot snapshot = snapshot("assistant", 1L);

        StartAgentRunCommand firstCommand = rootCommand(
                uniqueId("interrupted-root"),
                conversationId,
                42L,
                7L,
                snapshot,
                Instant.now().plusSeconds(300),
                null,
                null,
                null);
        StartedAgentRun first = await(coordinator.start(firstCommand));
        assertThat(runMapper.update(null, new LambdaUpdateWrapper<AgentRun>()
                .eq(AgentRun::getRunId, first.runId())
                .set(AgentRun::getStatus, "CANCELLED"))).isEqualTo(1);

        StartAgentRunCommand recoveryCommand = rootCommand(
                uniqueId("recovery-root"),
                conversationId,
                42L,
                7L,
                snapshot,
                Instant.now().plusSeconds(300),
                null,
                null,
                null);
        StartedAgentRun recovery = await(coordinator.start(recoveryCommand));

        assertThat(recovery.agentStateSessionId())
                .isEqualTo(AgentStateSessionIds.recoveryGeneration(
                        conversationId, "assistant", recovery.runId()))
                .startsWith("afv-root:")
                .isNotEqualTo(first.agentStateSessionId());

        assertThat(runMapper.update(null, new LambdaUpdateWrapper<AgentRun>()
                .eq(AgentRun::getRunId, recovery.runId())
                .set(AgentRun::getStatus, "COMPLETED"))).isEqualTo(1);
        StartAgentRunCommand nextTurnCommand = rootCommand(
                uniqueId("next-turn-root"),
                conversationId,
                42L,
                7L,
                snapshot,
                Instant.now().plusSeconds(300),
                null,
                null,
                null);
        StartedAgentRun nextTurn = await(coordinator.start(nextTurnCommand));

        assertThat(nextTurn.agentStateSessionId())
                .isEqualTo(recovery.agentStateSessionId());
    }

    @Test
    void admitsChildFromLockedParentAndMakesDuplicateAdmissionIdempotent() {
        String conversationId = createConversation(42L, 7L);
        AgentKernelSnapshot rootSnapshot = snapshot("root-agent", 1L);
        Instant rootDeadline = Instant.now().plus(Duration.ofMinutes(15));
        StartAgentRunCommand rootCommand = rootCommand(
                uniqueId("root"), conversationId, 42L, 7L,
                rootSnapshot, rootDeadline, null, null, null);
        StartedAgentRun root = await(coordinator.start(rootCommand));

        AgentKernelSnapshot childSnapshot = snapshot("asset-image-gen", 3L);
        Instant childDeadline = root.deadline().minusSeconds(60);
        StartChildAgentRunCommand childCommand = childCommand(
                uniqueId("child"), root.runId(), "tool-call-1",
                root.ownerInstanceId(), root.ownerEpoch(), "Asset image agent",
                "asset_image_gen", childSnapshot, childDeadline);

        ChildRunAdmission admitted = await(coordinator.startChild(childCommand));

        assertThat(admitted.created()).isTrue();
        assertThat(admitted.run().initialMessageOrder()).isEqualTo(2L);
        AgentRun persistedChild = run(childCommand.childRunId());
        assertThat(persistedChild.getConversationId()).isEqualTo(conversationId);
        assertThat(persistedChild.getUserId()).isEqualTo(42L);
        assertThat(persistedChild.getProjectId()).isEqualTo(7L);
        assertThat(persistedChild.getParentRunId()).isEqualTo(root.runId());
        assertThat(persistedChild.getParentToolCallId()).isEqualTo("tool-call-1");
        assertThat(persistedChild.getAgentName()).isEqualTo("Asset image agent");
        assertThat(persistedChild.getDeadlineAt())
                .isBeforeOrEqualTo(run(root.runId()).getDeadlineAt());
        assertThat(persistedChild.getAgentStateSessionId())
                .startsWith("afv-child:")
                .isNotEqualTo(root.agentStateSessionId());
        assertConversationCounters(conversationId, 2, 3L);

        StartChildAgentRunCommand retryWithNewCandidateId = childCommand(
                uniqueId("retry-candidate"), root.runId(), "tool-call-1",
                root.ownerInstanceId(), root.ownerEpoch(), "Asset image agent",
                "asset_image_gen", childSnapshot, childDeadline);
        ChildRunAdmission duplicate = await(coordinator.startChild(retryWithNewCandidateId));
        assertThat(duplicate.created()).isFalse();
        assertThat(duplicate.run().runId()).isEqualTo(childCommand.childRunId());
        assertThat(duplicate.run().initialMessageOrder()).isEqualTo(2L);
        assertConversationCounters(conversationId, 2, 3L);
        assertThat(messages(childCommand.childRunId())).hasSize(1);

        StartChildAgentRunCommand identityConflict = childCommand(
                childCommand.childRunId(), root.runId(), "tool-call-1",
                root.ownerInstanceId(), root.ownerEpoch(), "Different agent",
                "asset_image_gen", childSnapshot, childDeadline);
        assertThatThrownBy(() -> await(coordinator.startChild(identityConflict)))
                .isInstanceOf(ChildRunIdentityConflictException.class);

        StartChildAgentRunCommand afterParentDeadline = childCommand(
                uniqueId("late-child"), root.runId(), "tool-call-2",
                root.ownerInstanceId(), root.ownerEpoch(), "Late child",
                "asset_image_gen", childSnapshot, root.deadline().plusSeconds(1));
        assertThatThrownBy(() -> await(coordinator.startChild(afterParentDeadline)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("must not exceed parent");

        StartChildAgentRunCommand staleOwner = childCommand(
                uniqueId("stale-owner-child"), root.runId(), "tool-call-3",
                "stale-instance", root.ownerEpoch(), "Stale owner child",
                "asset_image_gen", childSnapshot, childDeadline);
        assertThatThrownBy(() -> await(coordinator.startChild(staleOwner)))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("所有权");

        assertThat(runMapper.update(null, new LambdaUpdateWrapper<AgentRun>()
                .eq(AgentRun::getRunId, root.runId())
                .set(AgentRun::getStatus, "CANCEL_REQUESTED"))).isEqualTo(1);
        StartChildAgentRunCommand cancelledParent = childCommand(
                uniqueId("cancelled-child"), root.runId(), "tool-call-4",
                root.ownerInstanceId(), root.ownerEpoch(), "Cancelled child",
                "asset_image_gen", childSnapshot, childDeadline);
        assertThatThrownBy(() -> await(coordinator.startChild(cancelledParent)))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("不接受子任务");
    }

    @Test
    void rechecksDatabaseTimeAfterWaitingForTheParentRowLock() throws Exception {
        String conversationId = createConversation(42L, 7L);
        AgentKernelSnapshot snapshot = snapshot("root-agent", 1L);
        StartedAgentRun root = await(coordinator.start(rootCommand(
                uniqueId("lease-root"), conversationId, 42L, 7L,
                snapshot, Instant.now().plusSeconds(300), null, null, null)));

        java.time.LocalDateTime leaseExpiry = runMapper.selectDatabaseNow().plusSeconds(1);
        assertThat(runMapper.update(null, new LambdaUpdateWrapper<AgentRun>()
                .eq(AgentRun::getRunId, root.runId())
                .set(AgentRun::getLeaseUntil, leaseExpiry))).isEqualTo(1);

        CountDownLatch parentLocked = new CountDownLatch(1);
        CountDownLatch releaseParent = new CountDownLatch(1);
        ExecutorService lockHolder = Executors.newSingleThreadExecutor();
        Future<?> lockFuture = lockHolder.submit(() -> transactionTemplate.execute(ignored -> {
            runMapper.selectByRunIdForUpdate(root.runId());
            parentLocked.countDown();
            try {
                if (!releaseParent.await(15, TimeUnit.SECONDS)) {
                    throw new IllegalStateException("Timed out waiting to release parent lock");
                }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Parent lock holder was interrupted", interrupted);
            }
            return null;
        }));
        try {
            assertThat(parentLocked.await(10, TimeUnit.SECONDS)).isTrue();
            StartChildAgentRunCommand child = childCommand(
                    uniqueId("expired-child"), root.runId(), "lease-tool-call",
                    root.ownerInstanceId(), root.ownerEpoch(), "Expired child",
                    "asset_image_gen", snapshot, root.deadline().minusSeconds(1));
            CompletableFuture<ChildRunAdmission> admission =
                    coordinator.startChild(child).toFuture();

            org.awaitility.Awaitility.await().atMost(Duration.ofSeconds(10))
                    .until(() -> runMapper.selectDatabaseNow().isAfter(leaseExpiry));
            releaseParent.countDown();

            assertThatThrownBy(() -> admission.get(15, TimeUnit.SECONDS))
                    .isInstanceOf(ExecutionException.class)
                    .hasCauseInstanceOf(BusinessException.class)
                    .hasRootCauseMessage("父 Agent 运行租约已失效");
            assertThat(run(child.childRunId())).isNull();
            lockFuture.get(10, TimeUnit.SECONDS);
        } finally {
            releaseParent.countDown();
            lockHolder.shutdownNow();
            assertThat(lockHolder.awaitTermination(10, TimeUnit.SECONDS)).isTrue();
        }
    }

    private String createConversation(long userId, long projectId) {
        String conversationId = uniqueId("conversation");
        conversationService.createOrUpdate(
                conversationId, userId, projectId, "project", projectId,
                "assistant", "Run start test", "chat");
        return conversationId;
    }

    private StartAgentRunCommand rootCommand(
            String runId,
            String conversationId,
            long userId,
            long projectId,
            AgentKernelSnapshot snapshot,
            Instant deadline,
            String parentRunId,
            String parentToolCallId,
            String agentName) {
        return new StartAgentRunCommand(
                runId,
                conversationId,
                userId,
                projectId,
                "assistant",
                parentRunId,
                parentToolCallId,
                agentName,
                "session-" + runId,
                snapshot,
                "instance-a",
                Duration.ofSeconds(30),
                deadline,
                "hello",
                null);
    }

    private StartChildAgentRunCommand childCommand(
            String childRunId,
            String parentRunId,
            String parentToolCallId,
            String parentOwner,
            long parentEpoch,
            String agentName,
            String agentDefinitionStableKey,
            AgentKernelSnapshot snapshot,
            Instant deadline) {
        return new StartChildAgentRunCommand(
                childRunId,
                parentRunId,
                parentToolCallId,
                parentOwner,
                parentEpoch,
                agentName,
                agentDefinitionStableKey,
                snapshot,
                "instance-child",
                Duration.ofSeconds(30),
                deadline,
                "child request",
                null);
    }

    private AgentKernelSnapshot snapshot(String stableKey, long modelVersion) {
        AgentKernelSnapshotPayload payload = new AgentKernelSnapshotPayload(
                AgentKernelSnapshotPayload.CURRENT_SCHEMA_VERSION,
                stableKey,
                stableKey + " name",
                "test agent",
                "You are a test agent.",
                10,
                "model-config-1",
                modelVersion,
                "openai",
                "gpt-test",
                JsonNodeFactory.instance.objectNode().put("temperature", 0.2),
                List.of(new ToolManifestSnapshot(
                        "read_asset", "a".repeat(64), true, true, "test-v1")),
                "1.0.0-test");
        return new CanonicalAgentKernelSnapshotBuilder().build(payload);
    }

    private AgentRun run(String runId) {
        return runMapper.selectOne(new LambdaQueryWrapper<AgentRun>()
                .eq(AgentRun::getRunId, runId));
    }

    private List<AgentMessage> messages(String runId) {
        return messageMapper.selectList(new LambdaQueryWrapper<AgentMessage>()
                .eq(AgentMessage::getRunId, runId)
                .orderByAsc(AgentMessage::getMessageOrder));
    }

    private void assertConversationCounters(
            String conversationId, int messageCount, long nextMessageOrder) {
        AgentConversation conversation = conversationMapper.selectOne(
                new LambdaQueryWrapper<AgentConversation>()
                        .eq(AgentConversation::getConversationId, conversationId));
        assertThat(conversation.getMessageCount()).isEqualTo(messageCount);
        assertThat(conversation.getNextMessageOrder()).isEqualTo(nextMessageOrder);
    }

    private <T> T await(reactor.core.publisher.Mono<T> result) {
        // Blocking is restricted to this integration-test boundary.
        return result.block(Duration.ofSeconds(15));
    }

    private static String uniqueId(String prefix) {
        return prefix + '-' + UUID.randomUUID().toString().replace("-", "");
    }
}
