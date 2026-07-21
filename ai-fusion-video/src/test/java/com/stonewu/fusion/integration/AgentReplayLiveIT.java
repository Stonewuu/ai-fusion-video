package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentTerminalOutputType;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import com.stonewu.fusion.service.ai.run.AgentEventJournal;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.AgentRunRedisSignalService;
import com.stonewu.fusion.service.ai.run.AgentRunReplayService;
import com.stonewu.fusion.service.ai.run.RunTerminalCoordinator;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.stream.LongStream;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@Testcontainers(disabledWithoutDocker = false)
class AgentReplayLiveIT {

    private static final int REDIS_PORT = 6379;
    private static final int PAGED_HISTORY_SIZE = 260;

    @Container
    private static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video")
            .withUsername("afv")
            .withPassword("afv-test");

    @Container
    private static final GenericContainer<?> REDIS = new GenericContainer<>(
            DockerImageName.parse("redis:7.4-alpine"))
            .withExposedPorts(REDIS_PORT);

    @DynamicPropertySource
    static void configure(DynamicPropertyRegistry properties) {
        properties.add("spring.datasource.url", MYSQL::getJdbcUrl);
        properties.add("spring.datasource.username", MYSQL::getUsername);
        properties.add("spring.datasource.password", MYSQL::getPassword);
        properties.add("spring.datasource.hikari.maximum-pool-size", () -> 16);
        properties.add("spring.datasource.hikari.minimum-idle", () -> 0);
        properties.add("spring.data.redis.host", REDIS::getHost);
        properties.add("spring.data.redis.port", () -> REDIS.getMappedPort(REDIS_PORT));
        properties.add("fusion.agentscope.v2.state.mode", () -> "in-memory");
    }

    @Autowired
    private AgentRunReplayService replayService;

    @Autowired
    private AgentRunRedisSignalService signals;

    @Autowired
    private AgentEventJournal journal;

    @Autowired
    private RunTerminalCoordinator terminals;

    @Autowired
    private AgentRunCoordinator runCoordinator;

    @Autowired
    private AgentConversationService conversationService;

    @Test
    void terminalHistoryReplaysFromMysqlAcrossPagesAndHonorsCursor() {
        StartedAgentRun run = startRoot("paged-terminal");
        for (int sequence = 1; sequence <= PAGED_HISTORY_SIZE; sequence++) {
            AgentEventEnvelope event = sequence == 130
                    ? rawCheckpoint(sequence)
                    : contentEvent(sequence);
            assertThat(await(journal.appendOwned(
                    run.runId(),
                    run.ownerInstanceId(),
                    run.ownerEpoch(),
                    event)))
                    .isPresent();
        }
        CommittedAgentEvent terminal = await(terminals.terminateOwned(
                completed(run), run.ownerInstanceId(), run.ownerEpoch()))
                .orElseThrow();
        assertThat(terminal.sequence()).isEqualTo(PAGED_HISTORY_SIZE + 1L);

        List<CommittedAgentEvent> full = replayService
                .replayThenLive(run.runId(), 0)
                .collectList()
                .block(Duration.ofSeconds(30));

        assertThat(full).isNotNull();
        List<Long> expected = new ArrayList<>(LongStream
                .rangeClosed(1, PAGED_HISTORY_SIZE + 1L)
                .boxed()
                .toList());
        expected.remove(Long.valueOf(130L));
        assertThat(full)
                .extracting(CommittedAgentEvent::sequence)
                .containsExactlyElementsOf(expected);
        assertThat(full.getLast().outputType()).isEqualTo("DONE");

        List<CommittedAgentEvent> resumed = replayService
                .replayThenLive(run.runId(), PAGED_HISTORY_SIZE - 2L)
                .collectList()
                .block(Duration.ofSeconds(15));
        assertThat(resumed)
                .extracting(CommittedAgentEvent::sequence)
                .containsExactly(
                        PAGED_HISTORY_SIZE - 1L,
                        (long) PAGED_HISTORY_SIZE,
                        PAGED_HISTORY_SIZE + 1L);

        assertThat(replayService
                .replayThenLive(run.runId(), terminal.sequence())
                .collectList()
                .block(Duration.ofSeconds(15)))
                .isEmpty();
    }

    @Test
    void replayLiveBoundaryIgnoresDuplicateAndReorderedWakeupHints()
            throws Exception {
        StartedAgentRun run = startRoot("live-boundary");
        CommittedAgentEvent replayed = append(run, contentEvent(1));
        assertThat(replayed.sequence()).isEqualTo(1L);

        List<Long> delivered = new CopyOnWriteArrayList<>();
        CountDownLatch nonTerminalDelivered = new CountDownLatch(2);
        CompletableFuture<Void> completion = replayService
                .replayThenLive(run.runId(), 0)
                .doOnNext(event -> {
                    delivered.add(event.sequence());
                    if (event.sequence() <= 2) {
                        nonTerminalDelivered.countDown();
                    }
                })
                .then()
                .toFuture();

        CommittedAgentEvent live = append(run, contentEvent(2));
        await(signals.publishWakeup(run.runId(), live.sequence()));
        await(signals.publishWakeup(run.runId(), replayed.sequence()));
        await(signals.publishWakeup(run.runId(), live.sequence()));
        await(signals.publishWakeup(run.runId(), 999));

        assertThat(nonTerminalDelivered.await(15, TimeUnit.SECONDS)).isTrue();
        assertThat(completion.isDone()).isFalse();

        CommittedAgentEvent terminal = await(terminals.terminateOwned(
                completed(run), run.ownerInstanceId(), run.ownerEpoch()))
                .orElseThrow();
        await(signals.publishWakeup(run.runId(), live.sequence()));
        await(signals.publishWakeup(run.runId(), terminal.sequence()));
        await(signals.publishWakeup(run.runId(), terminal.sequence()));

        completion.get(20, TimeUnit.SECONDS);
        assertThat(delivered).containsExactly(1L, 2L, 3L);
    }

    @Test
    void periodicMysqlPollRecoversWhenEveryRedisWakeupIsLost() throws Exception {
        StartedAgentRun run = startRoot("lost-wakeups");
        CompletableFuture<List<CommittedAgentEvent>> replayed = replayService
                .replayThenLive(run.runId(), 0)
                .collectList()
                .toFuture();

        append(run, contentEvent(1));
        await(terminals.terminateOwned(
                completed(run), run.ownerInstanceId(), run.ownerEpoch()))
                .orElseThrow();

        List<CommittedAgentEvent> events = replayed.get(20, TimeUnit.SECONDS);
        assertThat(events)
                .extracting(CommittedAgentEvent::sequence)
                .containsExactly(1L, 2L);
        assertThat(events.getLast().outputType()).isEqualTo("DONE");
    }

    private CommittedAgentEvent append(
            StartedAgentRun run, AgentEventEnvelope event) {
        return await(journal.appendOwned(
                run.runId(),
                run.ownerInstanceId(),
                run.ownerEpoch(),
                event)).orElseThrow();
    }

    private StartedAgentRun startRoot(String prefix) {
        long userId = 42L;
        long projectId = 7L;
        String conversationId = uniqueId("conversation-" + prefix);
        conversationService.createOrUpdate(
                conversationId,
                userId,
                projectId,
                "project",
                projectId,
                "assistant",
                "Replay test",
                "chat");
        String runId = uniqueId("run-" + prefix);
        return await(runCoordinator.start(new StartAgentRunCommand(
                runId,
                conversationId,
                userId,
                projectId,
                "assistant",
                null,
                null,
                null,
                "session-" + runId,
                snapshot(),
                "instance-a",
                Duration.ofMinutes(2),
                Instant.now().plus(Duration.ofMinutes(10)),
                "hello",
                null)));
    }

    private AgentKernelSnapshot snapshot() {
        return new CanonicalAgentKernelSnapshotBuilder().build(
                new AgentKernelSnapshotPayload(
                        1,
                        "assistant",
                        "Assistant",
                        "Replay test agent",
                        "You are a replay test agent.",
                        10,
                        "model-config-1",
                        1L,
                        "openai",
                        "gpt-test",
                        JsonNodeFactory.instance.objectNode().put("temperature", 0.2),
                        List.of(),
                        "0.6.3-test"));
    }

    private AgentEventEnvelope contentEvent(int index) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", "CONTENT")
                .put("content", "chunk-" + index)
                .put("finished", false);
        return envelope(
                uniqueId("content-" + index),
                "TEXT_BLOCK_DELTA",
                "CONTENT",
                payload);
    }

    private AgentEventEnvelope rawCheckpoint(int index) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("checkpoint", index);
        return envelope(
                uniqueId("raw-" + index),
                "INTERNAL_CHECKPOINT",
                null,
                payload);
    }

    private RunTerminalRequest completed(StartedAgentRun run) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", "DONE")
                .put("finished", true);
        return new RunTerminalRequest(
                run.runId(),
                new StateStoreSlot("42", run.agentStateSessionId()),
                Set.of(AgentRunStatus.RUNNING),
                AgentRunStatus.COMPLETED,
                AgentTerminalOutputType.DONE,
                null,
                null,
                envelope(uniqueId("done"), "AGENT_END", "DONE", payload));
    }

    private AgentEventEnvelope envelope(
            String rawEventId,
            String rawEventType,
            String outputType,
            ObjectNode payload) {
        return new AgentEventEnvelope(
                rawEventId,
                rawEventType,
                "assistant",
                null,
                null,
                null,
                null,
                null,
                outputType,
                payload,
                Instant.now());
    }

    private <T> T await(Mono<T> value) {
        // Blocking is restricted to this integration-test boundary.
        return value.block(Duration.ofSeconds(30));
    }

    private static String uniqueId(String prefix) {
        String boundedPrefix = prefix.length() <= 24
                ? prefix
                : prefix.substring(0, 24);
        return boundedPrefix + '-'
                + UUID.randomUUID().toString().replace("-", "");
    }
}
