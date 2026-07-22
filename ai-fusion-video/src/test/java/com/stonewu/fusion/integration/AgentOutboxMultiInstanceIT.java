package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.AgentEventEnvelopeSanitizer;
import com.stonewu.fusion.service.ai.run.AgentEventJournal;
import com.stonewu.fusion.service.ai.run.AgentEventOutboxPublisher;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.AgentRunRedisSignalService;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.transaction.PlatformTransactionManager;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@Testcontainers(disabledWithoutDocker = false)
class AgentOutboxMultiInstanceIT {

    private static final int REDIS_PORT = 6379;

    @Container
    private static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video")
            .withUsername("afv")
            .withPassword("afv-test");

    @Container
    private static final GenericContainer<?> REDIS = new GenericContainer<>(
            DockerImageName.parse("redis:7.4.5-alpine"))
            .withExposedPorts(REDIS_PORT);

    @DynamicPropertySource
    static void configure(DynamicPropertyRegistry properties) {
        properties.add("spring.datasource.url", MYSQL::getJdbcUrl);
        properties.add("spring.datasource.username", MYSQL::getUsername);
        properties.add("spring.datasource.password", MYSQL::getPassword);
        properties.add("spring.datasource.hikari.maximum-pool-size", () -> 32);
        properties.add("spring.datasource.hikari.minimum-idle", () -> 0);
        properties.add("spring.data.redis.host", REDIS::getHost);
        properties.add("spring.data.redis.port", () -> REDIS.getMappedPort(REDIS_PORT));
        properties.add("fusion.agentscope.v2.state.mode", () -> "in-memory");
    }

    @Autowired
    private AgentEventOutboxPublisher publisher;

    @Autowired
    private AgentRunRedisSignalService signals;

    @Autowired
    private AgentEventJournal journal;

    @Autowired
    private AgentRunCoordinator runCoordinator;

    @Autowired
    private AgentConversationService conversationService;

    @Autowired
    private AgentEventMapper eventMapper;

    @Autowired
    private AgentRunMapper runMapper;

    @Autowired
    private PlatformTransactionManager transactionManager;

    @Autowired
    private AgentEventEnvelopeSanitizer sanitizer;

    @Autowired
    private AgentRuntimeSchedulers schedulers;

    @BeforeEach
    void clearOutbox() {
        eventMapper.delete(null);
    }

    @Test
    void twoNodesClaimDisjointRowsAndPublishEveryCommittedSequence()
            throws Exception {
        StartedAgentRun run = startRoot("outbox-two-node");
        appendEvents(run, 8);
        Flux<Long> activeWakeups = await(signals.wakeupsWhenSubscribed(run.runId()));
        CompletableFuture<List<Long>> received = activeWakeups
                .take(8)
                .collectList()
                .timeout(Duration.ofSeconds(15))
                .toFuture();

        await(Mono.when(
                publisher.publishBatch("node-a", 4),
                publisher.publishBatch("node-b", 4)));

        List<Long> sequences = received.get(20, TimeUnit.SECONDS);
        sequences.sort(Comparator.naturalOrder());
        assertThat(sequences).containsExactly(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L);
        assertThat(events(run.runId()))
                .hasSize(8)
                .allSatisfy(event -> {
                    assertThat(event.getPublishStatus()).isEqualTo("PUBLISHED");
                    assertThat(event.getPublishAttempts()).isEqualTo(1);
                    assertThat(event.getPublishClaimOwner()).isNull();
                    assertThat(event.getRedisPublishedAt()).isNotNull();
                });
    }

    @Test
    void expiredClaimIsTakenOverAndOldOwnerCannotAcknowledge() throws Exception {
        StartedAgentRun run = startRoot("outbox-expired-claim");
        AgentEvent event = appendEvents(run, 1).getFirst();
        assertThat(eventMapper.update(null, new LambdaUpdateWrapper<AgentEvent>()
                .eq(AgentEvent::getId, event.getId())
                .set(AgentEvent::getPublishStatus, "CLAIMED")
                .set(AgentEvent::getPublishClaimOwner, "old-owner-token")
                .set(AgentEvent::getPublishClaimUntil,
                        runMapper.selectDatabaseNow().minusSeconds(1))
                .set(AgentEvent::getNextPublishAttemptAt, null)
                .set(AgentEvent::getPublishAttempts, 1)))
                .isEqualTo(1);
        assertThat(eventMapper.markPublished(event.getId(), "old-owner-token"))
                .isZero();

        Flux<Long> activeWakeups = await(signals.wakeupsWhenSubscribed(run.runId()));
        CompletableFuture<Long> received = activeWakeups.next()
                .timeout(Duration.ofSeconds(15))
                .toFuture();
        await(publisher.publishBatch("new-node", 1));

        assertThat(received.get(20, TimeUnit.SECONDS)).isEqualTo(1L);
        AgentEvent persisted = requireEvent(event.getId());
        assertThat(persisted.getPublishStatus()).isEqualTo("PUBLISHED");
        assertThat(persisted.getPublishAttempts()).isEqualTo(2);
        assertThat(persisted.getPublishClaimOwner()).isNull();
    }

    @Test
    void duplicateWakeupAfterCrashWindowRemainsSafeAndIdempotent() throws Exception {
        StartedAgentRun run = startRoot("outbox-duplicate-publish");
        AgentEvent event = appendEvents(run, 1).getFirst();
        Flux<Long> activeWakeups = await(signals.wakeupsWhenSubscribed(run.runId()));
        CompletableFuture<List<Long>> received = activeWakeups
                .take(2)
                .collectList()
                .timeout(Duration.ofSeconds(15))
                .toFuture();

        await(publisher.publishBatch("node-first", 1));
        assertThat(eventMapper.update(null, new LambdaUpdateWrapper<AgentEvent>()
                .eq(AgentEvent::getId, event.getId())
                .set(AgentEvent::getPublishStatus, "PENDING")
                .set(AgentEvent::getRedisPublishedAt, null)
                .set(AgentEvent::getNextPublishAttemptAt,
                        runMapper.selectDatabaseNow())))
                .isEqualTo(1);
        await(publisher.publishBatch("node-after-crash", 1));

        assertThat(received.get(20, TimeUnit.SECONDS)).containsExactly(1L, 1L);
        AgentEvent persisted = requireEvent(event.getId());
        assertThat(persisted.getPublishStatus()).isEqualTo("PUBLISHED");
        assertThat(persisted.getPublishAttempts()).isEqualTo(2);
        assertThat(events(run.runId())).hasSize(1);
    }

    @Test
    void poisonEventIsReleasedWithBoundedBackoffAndSanitizedError() {
        StartedAgentRun run = startRoot("outbox-poison");
        AgentEvent event = appendEvents(run, 1).getFirst();
        AgentRunRedisSignalService failingSignals = mock(AgentRunRedisSignalService.class);
        when(failingSignals.publishWakeup(anyString(), anyLong()))
                .thenReturn(Mono.error(new IllegalStateException(
                        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz")));
        AgentEventOutboxPublisher failingPublisher = new AgentEventOutboxPublisher(
                eventMapper,
                runMapper,
                transactionManager,
                failingSignals,
                sanitizer,
                schedulers);
        LocalDateTime before = runMapper.selectDatabaseNow();

        await(failingPublisher.publishBatch("poison-node", 1));

        LocalDateTime after = runMapper.selectDatabaseNow();
        AgentEvent persisted = requireEvent(event.getId());
        assertThat(persisted.getPublishStatus()).isEqualTo("PENDING");
        assertThat(persisted.getPublishAttempts()).isEqualTo(1);
        assertThat(persisted.getPublishClaimOwner()).isNull();
        assertThat(persisted.getLastPublishError()).isEqualTo("[SECRET_REDACTED]");
        assertThat(persisted.getNextPublishAttemptAt())
                .isAfterOrEqualTo(before.plusNanos(100_000_000L))
                .isBeforeOrEqualTo(after.plusSeconds(30));
        assertThat(eventMapper.markPublished(event.getId(), "poison-node"))
                .isZero();
    }

    private List<AgentEvent> appendEvents(StartedAgentRun run, int count) {
        List<AgentEvent> inserted = new ArrayList<>(count);
        for (int index = 1; index <= count; index++) {
            String rawEventId = uniqueId("outbox-event-" + index);
            ObjectNode payload = JsonNodeFactory.instance.objectNode()
                    .put("outputType", "CONTENT")
                    .put("content", "chunk-" + index)
                    .put("finished", false);
            AgentEventEnvelope envelope = new AgentEventEnvelope(
                    rawEventId,
                    "TEXT_BLOCK_DELTA",
                    "assistant",
                    null,
                    null,
                    null,
                    null,
                    null,
                    "CONTENT",
                    payload,
                    Instant.now());
            assertThat(await(journal.appendOwned(
                    run.runId(),
                    run.ownerInstanceId(),
                    run.ownerEpoch(),
                    envelope)))
                    .isPresent();
        }
        inserted.addAll(events(run.runId()));
        return inserted;
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
                "Outbox test",
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
                        AgentKernelSnapshotPayload.CURRENT_SCHEMA_VERSION,
                        "assistant",
                        "Assistant",
                        "Outbox test agent",
                        "You are an outbox test agent.",
                        10,
                        "model-config-1",
                        1L,
                        "openai",
                        "gpt-test",
                        JsonNodeFactory.instance.objectNode().put("temperature", 0.2),
                        List.of(),
                        "0.6.3-test"));
    }

    private List<AgentEvent> events(String runId) {
        return eventMapper.selectList(new LambdaQueryWrapper<AgentEvent>()
                .eq(AgentEvent::getRunId, runId)
                .orderByAsc(AgentEvent::getSequenceNo));
    }

    private AgentEvent requireEvent(long eventId) {
        AgentEvent event = eventMapper.selectById(eventId);
        assertThat(event).isNotNull();
        return event;
    }

    private <T> T await(Mono<T> value) {
        // Blocking is restricted to this integration-test boundary.
        return value.block(Duration.ofSeconds(30));
    }

    private static String uniqueId(String prefix) {
        String boundedPrefix = prefix.length() <= 24 ? prefix : prefix.substring(0, 24);
        return boundedPrefix + '-' + UUID.randomUUID().toString().replace("-", "");
    }
}
