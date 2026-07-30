package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentTerminalOutputType;
import com.stonewu.fusion.integration.support.AgentRuntimeContainers;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import com.stonewu.fusion.service.ai.run.AgentEventOutboxPublisher;
import com.stonewu.fusion.service.ai.run.AgentExecution;
import com.stonewu.fusion.service.ai.run.AgentMessageProjectionService;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.AgentRunReconciliationService;
import com.stonewu.fusion.service.ai.run.AgentRunReplayService;
import com.stonewu.fusion.service.ai.run.AgentWaitingStatePort;
import com.stonewu.fusion.service.ai.run.MySqlAgentEventJournal;
import com.stonewu.fusion.service.ai.run.OwnedExecutionRegistry;
import com.stonewu.fusion.service.ai.run.RunLeaseGuard;
import com.stonewu.fusion.service.ai.run.RunTerminalCoordinator;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import com.stonewu.fusion.service.ai.run.model.ExecutionStopReason;
import com.stonewu.fusion.service.ai.run.model.PendingConfirmation;
import com.stonewu.fusion.service.ai.run.model.ResumeConfirmationCommand;
import com.stonewu.fusion.service.ai.run.model.ResumedAgentRun;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import com.stonewu.fusion.service.ai.run.model.WaitingCheckpoint;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.transaction.support.TransactionTemplate;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.awaitility.Awaitility.await;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
@Testcontainers(disabledWithoutDocker = false)
class AgentDurableRuntimeMultiInstanceIT {

    private static final long USER_ID = 42L;
    private static final long PROJECT_ID = 77L;
    private static final String NODE_A = "node-a";
    private static final String NODE_B = "node-b";
    private static final String PENDING_TOOLS = """
            [{"toolCallId":"tool-a","toolName":"generate_image",\
            "argumentsPreview":"{}"}]
            """;
    private static final String SUSPENDED_TOOLS =
            "[{\"id\":\"tool-a\",\"name\":\"generate_image\",\"input\":{},"
                    + "\"metadata\":{},\"state\":\"ASKING\"}]";

    @Container
    private static final MySQLContainer<?> MYSQL = AgentRuntimeContainers.mysql();

    @Container
    private static final GenericContainer<?> REDIS = AgentRuntimeContainers.redis();

    @DynamicPropertySource
    static void configure(DynamicPropertyRegistry properties) {
        AgentRuntimeContainers.registerSpringProperties(properties, MYSQL, REDIS);
        properties.add("spring.datasource.hikari.maximum-pool-size", () -> 24);
        properties.add("fusion.agentscope.v2.execution.instance-id", () -> NODE_A);
        properties.add(
                "fusion.agentscope.v2.execution.maintenance-initial-delay-ms",
                () -> "3600000");
    }

    @Autowired
    private AgentRunCoordinator coordinator;

    @Autowired
    private AgentWaitingStatePort waiting;

    @Autowired
    private RunTerminalCoordinator terminals;

    @Autowired
    private RunLeaseGuard leases;

    @Autowired
    private AgentRunReconciliationService reconciliation;

    @Autowired
    private AgentMessageProjectionService projection;

    @Autowired
    private AgentRunReplayService replay;

    @Autowired
    private AgentEventOutboxPublisher outbox;

    @Autowired
    private OwnedExecutionRegistry executions;

    @Autowired
    private AgentConversationService conversationService;

    @Autowired
    private AgentRunRepository runRepository;

    @Autowired
    private AgentEventRepository eventRepository;

    @Autowired
    private AgentRunMapper runMapper;

    @Autowired
    private AgentEventMapper eventMapper;

    @Autowired
    private AgentRuntimeSchedulers schedulers;

    @Autowired
    private TransactionTemplate transactions;

    private RuntimeNode nodeA;
    private RuntimeNode nodeB;

    @BeforeEach
    void createIndependentNodeAdapters() {
        nodeA = new RuntimeNode(
                NODE_A,
                new MySqlAgentEventJournal(eventRepository, schedulers));
        nodeB = new RuntimeNode(
                NODE_B,
                new MySqlAgentEventJournal(eventRepository, schedulers));
    }

    @Test
    void convergesAcrossOwnerTakeoverLostSignalAndOutboxClaimCrash() {
        StartedAgentRun started = nodeA.start();
        CommittedAgentEvent first = nodeA.append(started, 1L, content("from-a"));
        awaitMono(projection.projectThrough(started.runId(), first.sequence()));
        assertThat(run(started.runId()).getProjectedThroughSequence())
                .isEqualTo(first.sequence());

        PendingConfirmation candidate = new PendingConfirmation(
                "reply-1",
                Set.of("tool-a"),
                PENDING_TOOLS,
                SUSPENDED_TOOLS,
                started.deadline().minusSeconds(10));
        awaitMono(waiting.recordConfirmationCandidate(started.runId(), candidate));
        AgentRun beforeWaiting = run(started.runId());
        WaitingCheckpoint checkpoint = new WaitingCheckpoint(
                beforeWaiting.getAgentStateSessionId(),
                beforeWaiting.getKernelFingerprint(),
                beforeWaiting.getAgentDefinitionSnapshotJson(),
                beforeWaiting.getNextSequence() - 1);
        assertThat(awaitMono(waiting.enterWaitingConfirmation(
                started.runId(), started.ownerEpoch(), checkpoint))).isTrue();

        ResumedAgentRun resumed = awaitMono(waiting.resumeConfirmation(
                new ResumeConfirmationCommand(
                        started.runId(),
                        USER_ID,
                        candidate.replyId(),
                        candidate.decisionIds(),
                        Map.of("tool-a", true),
                        NODE_B,
                        Duration.ofSeconds(30))));
        assertThat(resumed.newOwnerEpoch()).isEqualTo(started.ownerEpoch() + 1);

        assertThat(awaitMono(nodeA.journal.appendOwned(
                started.runId(), NODE_A, started.ownerEpoch(), content("stale-a"))))
                .isEmpty();
        assertThat(awaitMono(terminals.terminateOwned(
                completedRequest(started), NODE_A, started.ownerEpoch())))
                .isEmpty();
        nodeB.append(started, resumed.newOwnerEpoch(), content("from-b"));

        AtomicReference<ExecutionStopReason> stopReason = new AtomicReference<>();
        AgentExecution nodeBExecution = new AgentExecution(
                started.runId(),
                NODE_B,
                resumed.newOwnerEpoch(),
                USER_ID,
                started.agentStateSessionId(),
                Flux.never(),
                reason -> Mono.fromRunnable(() -> stopReason.set(reason)),
                () -> { });
        awaitMono(executions.registerAndLaunch(
                nodeBExecution,
                started.deadline(),
                events -> events,
                Flux::then,
                ignored -> Mono.empty()));

        // Intentionally update only MySQL: no Redis cancellation message is published.
        transactions.executeWithoutResult(ignored ->
                runRepository.requestAuthorizedCancellationTree(
                        started.runId(), USER_ID));
        awaitMono(leases.heartbeat(
                started.runId(), NODE_B, resumed.newOwnerEpoch(), Duration.ofSeconds(30)));
        awaitMono(executions.awaitEmpty(Duration.ofSeconds(5)));
        assertThat(stopReason.get()).isEqualTo(ExecutionStopReason.CANCEL_REQUESTED);
        assertThat(run(started.runId()).getCancelAcknowledgedAt()).isNotNull();

        awaitMono(reconciliation.reconcileExpiredOwner(started.runId()));
        awaitMono(reconciliation.reconcileExpiredOwner(started.runId()));
        AgentRun terminalRun = run(started.runId());
        assertThat(terminalRun.getStatus()).isEqualTo(AgentRunStatus.CANCELLED.name());
        assertThat(terminalEvents(started.runId())).hasSize(1);

        awaitMono(projection.projectThrough(
                started.runId(), terminalRun.getTerminalSequence()));
        List<CommittedAgentEvent> replayed = nodeB.replayAll(started.runId());
        assertThat(replayed).isNotEmpty();
        assertThat(replayed.getLast().sequence())
                .isEqualTo(terminalRun.getTerminalSequence());
        assertThat(replayed.getLast().outputType())
                .isEqualTo(AgentTerminalOutputType.CANCELLED.name());

        AgentEvent crashedClaim = requiredEvents(started.runId()).getFirst();
        String staleClaim = "node-a:crashed";
        assertThat(eventMapper.update(null, new LambdaUpdateWrapper<AgentEvent>()
                .eq(AgentEvent::getId, crashedClaim.getId())
                .set(AgentEvent::getPublishStatus, "CLAIMED")
                .set(AgentEvent::getPublishClaimOwner, staleClaim)
                .set(AgentEvent::getPublishClaimUntil,
                        runMapper.selectDatabaseNow().minusSeconds(1))
                .set(AgentEvent::getPublishAttempts, 1)))
                .isEqualTo(1);
        assertThat(eventMapper.markPublished(crashedClaim.getId(), staleClaim)).isZero();
        awaitMono(outbox.publishBatch(NODE_B, 100));

        await().atMost(Duration.ofSeconds(10))
                .pollInterval(Duration.ofMillis(50))
                .untilAsserted(() -> {
                    AgentRun converged = run(started.runId());
                    assertThat(converged.getStatus())
                            .isEqualTo(AgentRunStatus.CANCELLED.name());
                    assertThat(terminalEvents(started.runId())).hasSize(1);
                    assertThat(converged.getProjectedThroughSequence())
                            .isEqualTo(converged.getTerminalSequence());
                    assertThat(converged.getProjectionCompletedAt()).isNotNull();
                    assertThat(eventMapper.countOutstandingPublish()).isZero();
                    assertThat(requiredEvents(started.runId()))
                            .allSatisfy(event -> {
                                assertThat(event.getPublishStatus())
                                        .isEqualTo("PUBLISHED");
                                assertThat(event.getPublishClaimOwner()).isNull();
                            });
                });
    }

    private RunTerminalRequest completedRequest(StartedAgentRun started) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", "DONE")
                .put("finished", true);
        return new RunTerminalRequest(
                started.runId(),
                new StateStoreSlot(
                        String.valueOf(USER_ID), started.agentStateSessionId()),
                Set.of(AgentRunStatus.RUNNING),
                AgentRunStatus.COMPLETED,
                AgentTerminalOutputType.DONE,
                null,
                null,
                new AgentEventEnvelope(
                        unique("terminal"),
                        "RUN_TERMINAL",
                        "main",
                        null,
                        null,
                        null,
                        null,
                        null,
                        "DONE",
                        payload,
                        Instant.now()));
    }

    private AgentEventEnvelope content(String value) {
        return new AgentEventEnvelope(
                unique("event"),
                "TEXT_BLOCK_DELTA",
                "main",
                null,
                null,
                null,
                null,
                null,
                "CONTENT",
                JsonNodeFactory.instance.objectNode()
                        .put("outputType", "CONTENT")
                        .put("delta", value)
                        .put("finished", false),
                Instant.now());
    }

    private AgentRun run(String runId) {
        return runMapper.selectOne(new LambdaQueryWrapper<AgentRun>()
                .eq(AgentRun::getRunId, runId));
    }

    private List<AgentEvent> requiredEvents(String runId) {
        return eventMapper.selectList(new LambdaQueryWrapper<AgentEvent>()
                        .eq(AgentEvent::getRunId, runId)
                        .eq(AgentEvent::getPublishRequired, true))
                .stream()
                .sorted(Comparator.comparing(AgentEvent::getSequenceNo))
                .toList();
    }

    private List<AgentEvent> terminalEvents(String runId) {
        return eventMapper.selectList(new LambdaQueryWrapper<AgentEvent>()
                .eq(AgentEvent::getRunId, runId)
                .eq(AgentEvent::getRawEventType, "RUN_TERMINAL"));
    }

    private AgentKernelSnapshot snapshot() {
        return new CanonicalAgentKernelSnapshotBuilder().build(
                new AgentKernelSnapshotPayload(
                        AgentKernelSnapshotPayload.CURRENT_SCHEMA_VERSION,
                        "assistant",
                        "assistant",
                        "multi-instance test",
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

    private <T> T awaitMono(Mono<T> publisher) {
        // Blocking is restricted to this integration-test boundary.
        return publisher.block(Duration.ofSeconds(30));
    }

    private static String unique(String prefix) {
        return prefix + '-' + UUID.randomUUID().toString().replace("-", "");
    }

    private final class RuntimeNode {

        private final String instanceId;
        private final MySqlAgentEventJournal journal;

        private RuntimeNode(String instanceId, MySqlAgentEventJournal journal) {
            this.instanceId = instanceId;
            this.journal = journal;
        }

        private StartedAgentRun start() {
            String conversationId = unique("conversation");
            conversationService.createOrUpdate(
                    conversationId,
                    USER_ID,
                    PROJECT_ID,
                    "project",
                    PROJECT_ID,
                    "assistant",
                    "multi-instance durability",
                    "chat");
            Instant deadline = Instant.now().plus(Duration.ofMinutes(2))
                    .truncatedTo(ChronoUnit.MILLIS);
            return awaitMono(coordinator.start(new StartAgentRunCommand(
                    unique("run"),
                    conversationId,
                    USER_ID,
                    PROJECT_ID,
                    "assistant",
                    null,
                    null,
                    null,
                    unique("session"),
                    snapshot(),
                    instanceId,
                    Duration.ofSeconds(30),
                    deadline,
                    "run",
                    null)));
        }

        private CommittedAgentEvent append(
                StartedAgentRun run,
                long ownerEpoch,
                AgentEventEnvelope event) {
            return awaitMono(journal.appendOwned(
                    run.runId(), instanceId, ownerEpoch, event)).orElseThrow();
        }

        private List<CommittedAgentEvent> replayAll(String runId) {
            return replay.replayThenLive(runId, 0)
                    .collectList()
                    .block(Duration.ofSeconds(30));
        }
    }
}
