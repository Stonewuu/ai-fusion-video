package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentTerminalOutputType;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.AgentWaitingStatePort;
import com.stonewu.fusion.service.ai.run.MySqlAgentEventJournal;
import com.stonewu.fusion.service.ai.run.RunTerminalCoordinator;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
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
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Set;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@Testcontainers(disabledWithoutDocker = false)
class AgentOwnedJournalTakeoverIT {

    private static final long USER_ID = 42L;
    private static final long PROJECT_ID = 77L;
    private static final String PENDING_TOOLS = """
            [{"toolCallId":"tool-a","toolName":"generate_image",\
            "argumentsPreview":"{}"}]
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
                "fusion.agentscope.v2.execution.instance-id", () -> "takeover-node-a");
        properties.add(
                "fusion.agentscope.v2.execution.maintenance-initial-delay-ms",
                () -> "3600000");
    }

    @Autowired
    private AgentEventRepository eventRepository;

    @Autowired
    private AgentRuntimeSchedulers schedulers;

    @Autowired
    private AgentWaitingStatePort waiting;

    @Autowired
    private RunTerminalCoordinator terminals;

    @Autowired
    private AgentRunCoordinator coordinator;

    @Autowired
    private AgentConversationService conversationService;

    @Autowired
    private AgentRunMapper runMapper;

    @Autowired
    private AgentEventMapper eventMapper;

    private MySqlAgentEventJournal nodeA;
    private MySqlAgentEventJournal nodeB;

    @BeforeEach
    void createNodeAdapters() {
        nodeA = new MySqlAgentEventJournal(eventRepository, schedulers);
        nodeB = new MySqlAgentEventJournal(eventRepository, schedulers);
    }

    @Test
    void oldOwnerCannotAppendOrTerminateAfterWaitingTakeover() {
        StartedAgentRun started = startRun();
        assertThat(await(nodeA.appendOwned(
                started.runId(), "node-a", 1L, event("before-wait"))))
                .isPresent();

        PendingConfirmation candidate = new PendingConfirmation(
                "reply-1",
                Set.of("tool-a"),
                PENDING_TOOLS,
                started.deadline().minusSeconds(10));
        await(waiting.recordConfirmationCandidate(started.runId(), candidate));
        AgentRun beforeWaiting = run(started.runId());
        WaitingCheckpoint checkpoint = new WaitingCheckpoint(
                beforeWaiting.getAgentStateSessionId(),
                beforeWaiting.getKernelFingerprint(),
                beforeWaiting.getAgentDefinitionSnapshotJson(),
                beforeWaiting.getNextSequence() - 1);
        assertThat(await(waiting.enterWaitingConfirmation(
                started.runId(), 1L, checkpoint))).isTrue();
        ResumedAgentRun resumed = await(waiting.resumeConfirmation(
                new ResumeConfirmationCommand(
                        started.runId(),
                        USER_ID,
                        candidate.replyId(),
                        candidate.decisionIds(),
                        "node-b",
                        Duration.ofSeconds(30))));
        assertThat(resumed.newOwnerEpoch()).isEqualTo(2L);

        assertThat(await(nodeA.appendOwned(
                started.runId(), "node-a", 1L, event("stale-owner"))))
                .isEmpty();
        assertThat(await(terminals.terminateOwned(
                completedRequest(started), "node-a", 1L))).isEmpty();

        assertThat(await(nodeB.appendOwned(
                started.runId(), "node-b", 2L, event("new-owner"))))
                .isPresent();
        assertThat(await(terminals.terminateOwned(
                completedRequest(started), "node-b", 2L))).isPresent();

        AgentRun completed = run(started.runId());
        assertThat(completed.getStatus()).isEqualTo(AgentRunStatus.COMPLETED.name());
        assertThat(completed.getOwnerInstanceId()).isEqualTo("node-b");
        assertThat(completed.getOwnerEpoch()).isEqualTo(2L);
        assertThat(eventMapper.selectList(new LambdaQueryWrapper<AgentEvent>()
                .eq(AgentEvent::getRunId, started.runId())
                .eq(AgentEvent::getRawEventType, "RUN_TERMINAL")))
                .hasSize(1);
    }

    private StartedAgentRun startRun() {
        String conversationId = unique("conversation");
        conversationService.createOrUpdate(
                conversationId,
                USER_ID,
                PROJECT_ID,
                "project",
                PROJECT_ID,
                "assistant",
                "takeover test",
                "chat");
        AgentKernelSnapshot snapshot = new CanonicalAgentKernelSnapshotBuilder().build(
                new AgentKernelSnapshotPayload(
                        1,
                        "assistant",
                        "assistant",
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
        Instant deadline = Instant.now().plusSeconds(90)
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
                unique("session"),
                snapshot,
                "node-a",
                Duration.ofSeconds(30),
                deadline,
                "take over",
                null)));
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

    private AgentEventEnvelope event(String id) {
        return new AgentEventEnvelope(
                id,
                "CUSTOM",
                "main",
                null,
                null,
                null,
                null,
                null,
                null,
                JsonNodeFactory.instance.objectNode().put("id", id),
                Instant.now());
    }

    private AgentRun run(String runId) {
        return runMapper.selectOne(new LambdaQueryWrapper<AgentRun>()
                .eq(AgentRun::getRunId, runId));
    }

    private <T> T await(Mono<T> publisher) {
        return publisher.block(Duration.ofSeconds(15));
    }

    private static String unique(String prefix) {
        return prefix + '-' + UUID.randomUUID().toString().replace("-", "");
    }
}
