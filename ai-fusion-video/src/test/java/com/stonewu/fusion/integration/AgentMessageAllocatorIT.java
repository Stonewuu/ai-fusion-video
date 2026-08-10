package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.run.AgentMessageAllocator;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.LongStream;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@Testcontainers(disabledWithoutDocker = false)
class AgentMessageAllocatorIT {

    private static final int CONCURRENT_WRITERS = 32;
    private static final int METADATA_WRITERS = 8;
    private static final int METADATA_UPDATES_PER_WRITER = 10;

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
        properties.add("spring.datasource.hikari.maximum-pool-size", () -> 40);
        properties.add("spring.datasource.hikari.minimum-idle", () -> 0);
    }

    @Autowired
    private AgentConversationService conversationService;

    @Autowired
    private AgentMessageAllocator messageAllocator;

    @Autowired
    private AgentMessageMapper messageMapper;

    @Autowired
    private AgentRunMapper runMapper;

    @Test
    void serializesConcurrentOrdersAndOnlyMessagesAdvanceConversationCounters() throws Exception {
        String conversationId = uniqueId("allocator");
        AgentConversation created = conversationService.createOrUpdate(
                conversationId, 42L, 7L, "project", 7L,
                "assistant", "Initial title", "chat");

        assertThat(created.getMessageCount()).isZero();
        assertThat(created.getNextMessageOrder()).isEqualTo(1L);
        assertThat(created.getLastMessageTime()).isNull();

        conversationService.createOrUpdate(
                conversationId, 42L, 7L, "project", 7L,
                "assistant", "Updated title", "chat");
        AgentConversation metadataOnlyUpdate =
                conversationService.getByConversationId(conversationId);
        assertThat(metadataOnlyUpdate.getMessageCount()).isZero();
        assertThat(metadataOnlyUpdate.getNextMessageOrder()).isEqualTo(1L);
        assertThat(metadataOnlyUpdate.getLastMessageTime()).isNull();

        CountDownLatch ready = new CountDownLatch(CONCURRENT_WRITERS + METADATA_WRITERS);
        CountDownLatch start = new CountDownLatch(1);
        ExecutorService writers = Executors.newFixedThreadPool(
                CONCURRENT_WRITERS + METADATA_WRITERS);
        List<Future<Long>> futures = new ArrayList<>(CONCURRENT_WRITERS);
        List<Future<?>> metadataFutures = new ArrayList<>(METADATA_WRITERS);
        try {
            for (int index = 0; index < CONCURRENT_WRITERS; index++) {
                int messageIndex = index;
                futures.add(writers.submit(() -> {
                    ready.countDown();
                    if (!start.await(30, TimeUnit.SECONDS)) {
                        throw new IllegalStateException("Concurrent writers did not start together");
                    }
                    return messageAllocator.append(conversationId, AgentMessage.builder()
                            .role("assistant")
                            .content("message-" + messageIndex)
                            .build());
                }));
            }
            for (int index = 0; index < METADATA_WRITERS; index++) {
                int metadataIndex = index;
                metadataFutures.add(writers.submit(() -> {
                    ready.countDown();
                    if (!start.await(30, TimeUnit.SECONDS)) {
                        throw new IllegalStateException("Metadata writers did not start together");
                    }
                    for (int update = 0; update < METADATA_UPDATES_PER_WRITER; update++) {
                        if ((metadataIndex + update) % 2 == 0) {
                            conversationService.createOrUpdate(
                                    conversationId, 42L, 7L, "project", 7L,
                                    "assistant", "title-" + metadataIndex + '-' + update, "chat");
                        } else {
                            conversationService.finish(conversationId, "running");
                        }
                    }
                    return null;
                }));
            }

            assertThat(ready.await(30, TimeUnit.SECONDS)).isTrue();
            start.countDown();

            List<Long> allocatedOrders = new ArrayList<>(CONCURRENT_WRITERS);
            for (Future<Long> future : futures) {
                allocatedOrders.add(future.get(30, TimeUnit.SECONDS));
            }
            for (Future<?> metadataFuture : metadataFutures) {
                metadataFuture.get(30, TimeUnit.SECONDS);
            }
            allocatedOrders.sort(Comparator.naturalOrder());
            assertThat(allocatedOrders)
                    .containsExactlyElementsOf(
                            LongStream.rangeClosed(1, CONCURRENT_WRITERS).boxed().toList());
        } finally {
            start.countDown();
            writers.shutdownNow();
            assertThat(writers.awaitTermination(30, TimeUnit.SECONDS)).isTrue();
        }

        AgentConversation persisted = conversationService.getByConversationId(conversationId);
        assertThat(persisted.getMessageCount()).isEqualTo(CONCURRENT_WRITERS);
        assertThat(persisted.getNextMessageOrder()).isEqualTo(CONCURRENT_WRITERS + 1L);
        assertThat(persisted.getLastMessageTime()).isNotNull();

        List<AgentMessage> messages = messageMapper.selectList(
                new LambdaQueryWrapper<AgentMessage>()
                        .eq(AgentMessage::getConversationId, conversationId)
                        .orderByAsc(AgentMessage::getMessageOrder));
        assertThat(messages).hasSize(CONCURRENT_WRITERS);
        assertThat(messages)
                .extracting(AgentMessage::getMessageOrder)
                .containsExactlyElementsOf(
                        LongStream.rangeClosed(1, CONCURRENT_WRITERS).boxed().toList());
    }

    @Test
    void mapsRootAndChildIdentityAndRequiredDeadlineThroughLockedRunReads() {
        LocalDateTime databaseNow = runMapper.selectDatabaseNow();
        String conversationId = uniqueId("run-conversation");
        String rootRunId = uniqueId("root");
        LocalDateTime rootDeadline = databaseNow.plusMinutes(10);

        AgentRun root = runningRun(
                rootRunId, conversationId, rootDeadline, databaseNow,
                null, null, null);
        assertThat(runMapper.insert(root)).isEqualTo(1);

        AgentRun persistedRoot = runMapper.selectByRunIdForUpdate(rootRunId);
        assertThat(persistedRoot.getDeadlineAt()).isEqualTo(rootDeadline);
        assertThat(persistedRoot.getParentRunId()).isNull();
        assertThat(persistedRoot.getParentToolCallId()).isNull();
        assertThat(persistedRoot.getAgentName()).isNull();

        String childRunId = uniqueId("child");
        String parentToolCallId = "tool-" + uniqueId("call");
        String agentName = "asset_image_gen";
        AgentRun child = runningRun(
                childRunId, conversationId, rootDeadline.minusMinutes(1), databaseNow,
                rootRunId, parentToolCallId, agentName);
        assertThat(runMapper.insert(child)).isEqualTo(1);

        AgentRun persistedChild = runMapper.selectByParentAndToolCallForUpdate(
                rootRunId, parentToolCallId);
        assertThat(persistedChild.getRunId()).isEqualTo(childRunId);
        assertThat(persistedChild.getParentRunId()).isEqualTo(rootRunId);
        assertThat(persistedChild.getParentToolCallId()).isEqualTo(parentToolCallId);
        assertThat(persistedChild.getAgentName()).isEqualTo(agentName);
        assertThat(persistedChild.getDeadlineAt()).isEqualTo(rootDeadline.minusMinutes(1));
        assertThat(runMapper.selectActiveChildren(rootRunId))
                .extracting(AgentRun::getRunId)
                .containsExactly(childRunId);
    }

    private static AgentRun runningRun(
            String runId,
            String conversationId,
            LocalDateTime deadlineAt,
            LocalDateTime databaseNow,
            String parentRunId,
            String parentToolCallId,
            String agentName) {
        return AgentRun.builder()
                .runId(runId)
                .conversationId(conversationId)
                .userId(42L)
                .agentType("assistant")
                .parentRunId(parentRunId)
                .parentToolCallId(parentToolCallId)
                .agentName(agentName)
                .kernelFingerprint("f".repeat(64))
                .agentDefinitionSnapshotJson("{}")
                .agentStateSessionId(runId + "-session")
                .status("RUNNING")
                .ownerInstanceId("instance-a")
                .ownerEpoch(1L)
                .leaseUntil(databaseNow.plus(Duration.ofSeconds(30)))
                .deadlineAt(deadlineAt)
                .startedAt(databaseNow)
                .build();
    }

    private static String uniqueId(String prefix) {
        return prefix + '-' + UUID.randomUUID().toString().replace("-", "");
    }
}
