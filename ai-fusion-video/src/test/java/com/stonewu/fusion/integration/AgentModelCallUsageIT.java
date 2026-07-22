package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AgentModelCallUsage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentModelCallStatus;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentRuntimeErrorCode;
import com.stonewu.fusion.enums.ai.AgentTerminalOutputType;
import com.stonewu.fusion.mapper.ai.AgentModelCallUsageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.repository.ai.AgentModelCallUsageRepository;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.RunTerminalCoordinator;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.ModelCallIdentityConflictException;
import com.stonewu.fusion.service.ai.run.model.NormalizedModelUsage;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashSet;
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
class AgentModelCallUsageIT {

    private static final int CONCURRENT_DUPLICATES = 16;

    @Container
    private static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video")
            .withUsername("afv")
            .withPassword("afv-test");

    @DynamicPropertySource
    static void configure(DynamicPropertyRegistry properties) {
        properties.add("spring.datasource.url", MYSQL::getJdbcUrl);
        properties.add("spring.datasource.username", MYSQL::getUsername);
        properties.add("spring.datasource.password", MYSQL::getPassword);
        properties.add("spring.datasource.hikari.maximum-pool-size", () -> 32);
        properties.add("spring.datasource.hikari.minimum-idle", () -> 0);
    }

    @Autowired
    private AgentModelCallUsageRepository repository;

    @Autowired
    private AgentModelCallUsageMapper usageMapper;

    @Autowired
    private AgentRunMapper runMapper;

    @Autowired
    private AgentRunCoordinator runCoordinator;

    @Autowired
    private RunTerminalCoordinator terminalCoordinator;

    @Autowired
    private AgentConversationService conversationService;

    @BeforeEach
    void clearUsageLedger() {
        usageMapper.delete(null);
    }

    @Test
    void startsTwoCallsAndDeduplicatesConcurrentIdenticalIdentity() throws Exception {
        StartedAgentRun run = startRoot("duplicate-start");
        ExecutorService callers = Executors.newFixedThreadPool(CONCURRENT_DUPLICATES);
        CountDownLatch ready = new CountDownLatch(CONCURRENT_DUPLICATES);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<?>> futures = new ArrayList<>();
        try {
            for (int index = 0; index < CONCURRENT_DUPLICATES; index++) {
                futures.add(callers.submit(() -> {
                    ready.countDown();
                    if (!start.await(20, TimeUnit.SECONDS)) {
                        throw new IllegalStateException(
                                "model-call starters did not become ready");
                    }
                    repository.startCall(
                            run.runId(), "call-1", "openai", "gpt-test");
                    return null;
                }));
            }
            assertThat(ready.await(20, TimeUnit.SECONDS)).isTrue();
            start.countDown();
            for (Future<?> future : futures) {
                future.get(30, TimeUnit.SECONDS);
            }
        } finally {
            start.countDown();
            callers.shutdownNow();
            assertThat(callers.awaitTermination(20, TimeUnit.SECONDS)).isTrue();
        }

        repository.startCall(run.runId(), "call-2", "openai", "gpt-test");
        assertThat(usages(run.runId()))
                .hasSize(2)
                .extracting(AgentModelCallUsage::getModelCallId)
                .containsExactly("call-1", "call-2");

        assertThatThrownBy(() -> repository.startCall(
                run.runId(), "call-1", "anthropic", "claude-test"))
                .isInstanceOf(ModelCallIdentityConflictException.class)
                .satisfies(error -> {
                    ModelCallIdentityConflictException conflict =
                            (ModelCallIdentityConflictException) error;
                    assertThat(conflict.getExistingProvider()).isEqualTo("openai");
                    assertThat(conflict.getRequestedProvider()).isEqualTo("anthropic");
                });
        assertThat(usages(run.runId())).hasSize(2);
    }

    @Test
    void completesExactlyOnceSanitizesUsageAndNeverClaimsStartedCalls() {
        StartedAgentRun run = startRoot("completion-cas");
        repository.startCall(run.runId(), "completed-call", "openai", "gpt-test");
        repository.startCall(run.runId(), "still-started", "openai", "gpt-test");

        NormalizedModelUsage usage = new NormalizedModelUsage(
                12L,
                7L,
                3L,
                2L,
                "{\"apiKey\":\"sk-secret-value\","
                        + "\"binary\":\"QUJDREVGR0hJSktMTU5PUA==\","
                        + "\"total\":19}");
        assertThat(repository.completeCall(
                run.runId(), "completed-call", usage)).isTrue();
        assertThat(repository.completeCall(
                run.runId(), "completed-call", usage)).isFalse();
        assertThat(repository.failCall(
                run.runId(), "completed-call", AgentModelCallStatus.FAILED)).isFalse();

        AgentModelCallUsage persisted = usage(run.runId(), "completed-call");
        assertThat(persisted.getStatus()).isEqualTo(AgentModelCallStatus.COMPLETED.name());
        assertThat(persisted.getInputTokens()).isEqualTo(12L);
        assertThat(persisted.getOutputTokens()).isEqualTo(7L);
        assertThat(persisted.getReasoningTokens()).isEqualTo(3L);
        assertThat(persisted.getCacheTokens()).isEqualTo(2L);
        assertThat(persisted.getUsageJson())
                .contains("[SECRET_REDACTED]", "[BINARY_REDACTED]")
                .doesNotContain("sk-secret-value", "QUJDREVGR0hJSktMTU5PUA==");

        List<AgentModelCallUsage> claimed = repository.claimSettlementBatch(
                "claim-completed-only", Duration.ofMinutes(1), 10);
        assertThat(claimed)
                .singleElement()
                .extracting(AgentModelCallUsage::getModelCallId)
                .isEqualTo("completed-call");
        assertThat(usage(run.runId(), "still-started").getSettlementStatus())
                .isEqualTo("PENDING");
    }

    @Test
    void completeFailCancelRaceProducesOneTerminalCallState() throws Exception {
        StartedAgentRun run = startRoot("call-terminal-race");
        repository.startCall(run.runId(), "raced-call", "openai", "gpt-test");
        CountDownLatch ready = new CountDownLatch(3);
        CountDownLatch start = new CountDownLatch(1);
        ExecutorService callers = Executors.newFixedThreadPool(3);
        List<Future<Boolean>> futures = new ArrayList<>();
        try {
            futures.add(callers.submit(() -> race(ready, start, () ->
                    repository.completeCall(
                            run.runId(), "raced-call",
                            NormalizedModelUsage.tokens(4, 5)))));
            futures.add(callers.submit(() -> race(ready, start, () ->
                    repository.failCall(
                            run.runId(), "raced-call", AgentModelCallStatus.FAILED))));
            futures.add(callers.submit(() -> race(ready, start, () ->
                    repository.failCall(
                            run.runId(), "raced-call", AgentModelCallStatus.CANCELLED))));
            assertThat(ready.await(20, TimeUnit.SECONDS)).isTrue();
            start.countDown();

            int winners = 0;
            for (Future<Boolean> future : futures) {
                if (future.get(20, TimeUnit.SECONDS)) {
                    winners++;
                }
            }
            assertThat(winners).isEqualTo(1);
        } finally {
            start.countDown();
            callers.shutdownNow();
            assertThat(callers.awaitTermination(20, TimeUnit.SECONDS)).isTrue();
        }

        AgentModelCallUsage persisted = usage(run.runId(), "raced-call");
        assertThat(persisted.getStatus()).isIn(
                AgentModelCallStatus.COMPLETED.name(),
                AgentModelCallStatus.FAILED.name(),
                AgentModelCallStatus.CANCELLED.name());
        assertThat(persisted.getFinishedAt()).isNotNull();
        assertThat(repository.claimSettlementBatch(
                "terminal-race-claim", Duration.ofMinutes(1), 10))
                .singleElement();
    }

    @Test
    void claimsDisjointBatchesAndFencesSettleRetryAndExpiredClaims()
            throws Exception {
        StartedAgentRun run = startRoot("claim-fencing");
        for (int index = 0; index < 8; index++) {
            String callId = "claim-call-" + index;
            repository.startCall(run.runId(), callId, "openai", "gpt-test");
            assertThat(repository.completeCall(
                    run.runId(), callId,
                    NormalizedModelUsage.tokens(index + 1L, index + 2L)))
                    .isTrue();
        }

        CountDownLatch start = new CountDownLatch(1);
        ExecutorService callers = Executors.newFixedThreadPool(2);
        List<AgentModelCallUsage> first;
        List<AgentModelCallUsage> second;
        try {
            Future<List<AgentModelCallUsage>> firstFuture = callers.submit(() -> {
                assertThat(start.await(20, TimeUnit.SECONDS)).isTrue();
                return repository.claimSettlementBatch(
                        "claim-owner-a", Duration.ofMinutes(2), 4);
            });
            Future<List<AgentModelCallUsage>> secondFuture = callers.submit(() -> {
                assertThat(start.await(20, TimeUnit.SECONDS)).isTrue();
                return repository.claimSettlementBatch(
                        "claim-owner-b", Duration.ofMinutes(2), 4);
            });
            start.countDown();
            first = firstFuture.get(30, TimeUnit.SECONDS);
            second = secondFuture.get(30, TimeUnit.SECONDS);
        } finally {
            start.countDown();
            callers.shutdownNow();
            assertThat(callers.awaitTermination(20, TimeUnit.SECONDS)).isTrue();
        }

        assertThat(first).hasSize(4);
        assertThat(second).hasSize(4);
        Set<Long> allIds = new HashSet<>();
        first.forEach(row -> assertThat(allIds.add(row.getId())).isTrue());
        second.forEach(row -> assertThat(allIds.add(row.getId())).isTrue());
        assertThat(allIds).hasSize(8);

        AgentModelCallUsage retry = first.getFirst();
        assertThat(repository.markSettled(
                retry.getId(), "wrong-owner", "settlement-wrong"))
                .isFalse();
        assertThat(repository.releaseSettlementForRetry(
                retry.getId(),
                "wrong-owner",
                Instant.now(),
                "wrong owner"))
                .isFalse();
        assertThat(repository.releaseSettlementForRetry(
                retry.getId(),
                "claim-owner-a",
                Instant.now().minusSeconds(10),
                "Authorization: Bearer abcdefghijklmnopqrstuvwxyz"))
                .isTrue();
        AgentModelCallUsage released = requireUsage(retry.getId());
        assertThat(released.getSettlementStatus()).isEqualTo("PENDING");
        assertThat(released.getSettlementClaimOwner()).isNull();
        assertThat(released.getLastSettlementError()).isEqualTo("[SECRET_REDACTED]");

        List<AgentModelCallUsage> reclaimed = repository.claimSettlementBatch(
                "claim-owner-c", Duration.ofMinutes(2), 1);
        assertThat(reclaimed)
                .singleElement()
                .extracting(AgentModelCallUsage::getId)
                .isEqualTo(retry.getId());
        assertThat(repository.markSettled(
                retry.getId(), "claim-owner-a", "stale-owner-settlement"))
                .isFalse();
        assertThat(repository.markSettled(
                retry.getId(), "claim-owner-c", "settlement-c"))
                .isTrue();

        AgentModelCallUsage expired = first.get(1);
        assertThat(usageMapper.update(null,
                new LambdaUpdateWrapper<AgentModelCallUsage>()
                        .eq(AgentModelCallUsage::getId, expired.getId())
                        .set(AgentModelCallUsage::getSettlementClaimUntil,
                                runMapper.selectDatabaseNow().minusSeconds(1))))
                .isEqualTo(1);
        List<AgentModelCallUsage> afterExpiry = repository.claimSettlementBatch(
                "claim-owner-d", Duration.ofMinutes(2), 1);
        assertThat(afterExpiry)
                .singleElement()
                .extracting(AgentModelCallUsage::getId)
                .isEqualTo(expired.getId());
        assertThat(repository.markSettled(
                expired.getId(), "claim-owner-a", "stale-expired-settlement"))
                .isFalse();
        assertThat(repository.markSettled(
                expired.getId(), "claim-owner-d", "settlement-d"))
                .isTrue();
    }

    @Test
    void marksRunSettledOnlyAfterTerminalAndEveryCallSettlement() {
        StartedAgentRun run = startRoot("run-settlement-order");
        repository.startCall(run.runId(), "call-1", "openai", "gpt-test");
        assertThat(repository.completeCall(
                run.runId(), "call-1", NormalizedModelUsage.tokens(1, 2)))
                .isTrue();
        AgentModelCallUsage first = repository.claimSettlementBatch(
                "order-owner-1", Duration.ofMinutes(2), 1).getFirst();
        assertThat(repository.markSettled(
                first.getId(), "order-owner-1", "settlement-1"))
                .isTrue();

        assertThat(repository.markRunUsageSettledIfAllCallsSettled(run.runId()))
                .isFalse();
        repository.startCall(run.runId(), "call-2", "openai", "gpt-test");
        assertThat(repository.completeCall(
                run.runId(), "call-2", NormalizedModelUsage.tokens(3, 4)))
                .isTrue();

        LocalDateTime databaseNow = runMapper.selectDatabaseNow();
        assertThat(runMapper.update(null, new LambdaUpdateWrapper<AgentRun>()
                .eq(AgentRun::getRunId, run.runId())
                .set(AgentRun::getStatus, AgentRunStatus.COMPLETED.name())
                .set(AgentRun::getFinishedAt, databaseNow)))
                .isEqualTo(1);
        assertThat(repository.markRunUsageSettledIfAllCallsSettled(run.runId()))
                .isFalse();

        AgentModelCallUsage second = repository.claimSettlementBatch(
                "order-owner-2", Duration.ofMinutes(2), 1).getFirst();
        assertThat(second.getModelCallId()).isEqualTo("call-2");
        assertThat(repository.markSettled(
                second.getId(), "order-owner-2", "settlement-2"))
                .isTrue();
        assertThat(repository.markRunUsageSettledIfAllCallsSettled(run.runId()))
                .isTrue();
        assertThat(repository.markRunUsageSettledIfAllCallsSettled(run.runId()))
                .isFalse();

        AgentRun persisted = requireRun(run.runId());
        assertThat(persisted.getUsageSettled()).isTrue();
        assertThat(persisted.getUsageSettledAt()).isNotNull();
        assertThatThrownBy(() -> repository.startCall(
                run.runId(), "call-after-terminal", "openai", "gpt-test"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("not accepting new model calls");
        assertThat(usages(run.runId())).hasSize(2);
    }

    @Test
    void terminalCasSweepsStrandedStartedCallsWithoutOverwritingCompletedUsage() {
        StartedAgentRun completedRun = startRoot("terminal-usage-sweep");
        repository.startCall(
                completedRun.runId(), "stranded-call", "openai", "gpt-test");
        repository.startCall(
                completedRun.runId(), "finished-call", "openai", "gpt-test");
        assertThat(repository.completeCall(
                completedRun.runId(), "finished-call",
                NormalizedModelUsage.tokens(8, 5)))
                .isTrue();

        assertThat(await(terminalCoordinator.terminateOwned(
                completedTerminal(completedRun),
                completedRun.ownerInstanceId(),
                completedRun.ownerEpoch())))
                .isPresent();
        assertThat(usage(completedRun.runId(), "stranded-call").getStatus())
                .isEqualTo(AgentModelCallStatus.FAILED.name());
        assertThat(usage(completedRun.runId(), "finished-call").getStatus())
                .isEqualTo(AgentModelCallStatus.COMPLETED.name());

        StartedAgentRun cancelledRun = startRoot("cancelled-usage-sweep");
        repository.startCall(
                cancelledRun.runId(), "cancelled-call", "openai", "gpt-test");
        assertThat(await(terminalCoordinator.terminateOwned(
                cancelledTerminal(cancelledRun),
                cancelledRun.ownerInstanceId(),
                cancelledRun.ownerEpoch())))
                .isPresent();
        assertThat(usage(cancelledRun.runId(), "cancelled-call").getStatus())
                .isEqualTo(AgentModelCallStatus.CANCELLED.name());
    }

    private boolean race(
            CountDownLatch ready,
            CountDownLatch start,
            ThrowingBooleanSupplier action) throws Exception {
        ready.countDown();
        if (!start.await(20, TimeUnit.SECONDS)) {
            throw new IllegalStateException("terminal callers did not become ready");
        }
        return action.getAsBoolean();
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
                "Usage test",
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
                        "Usage test agent",
                        "You are a usage test agent.",
                        10,
                        "model-config-1",
                        1L,
                        "openai",
                        "gpt-test",
                        JsonNodeFactory.instance.objectNode().put("temperature", 0.2),
                        List.of(),
                        "0.6.3-test"));
    }

    private RunTerminalRequest completedTerminal(StartedAgentRun run) {
        return terminalRequest(
                run,
                AgentRunStatus.COMPLETED,
                AgentTerminalOutputType.DONE,
                null,
                null,
                "AGENT_END");
    }

    private RunTerminalRequest cancelledTerminal(StartedAgentRun run) {
        return terminalRequest(
                run,
                AgentRunStatus.CANCELLED,
                AgentTerminalOutputType.CANCELLED,
                AgentRuntimeErrorCode.RUN_CANCELLED,
                "run cancelled",
                "REQUEST_STOP");
    }

    private RunTerminalRequest terminalRequest(
            StartedAgentRun run,
            AgentRunStatus terminalStatus,
            AgentTerminalOutputType outputType,
            AgentRuntimeErrorCode errorCode,
            String errorMessage,
            String rawEventType) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", outputType.name())
                .put("finished", true);
        if (errorCode != null) {
            payload.put("errorCode", errorCode.name());
        }
        if (errorMessage != null) {
            payload.put("error", errorMessage);
        }
        AgentEventEnvelope envelope = new AgentEventEnvelope(
                uniqueId("terminal-event"),
                rawEventType,
                "assistant",
                null,
                null,
                null,
                null,
                null,
                outputType.name(),
                payload,
                Instant.now());
        return new RunTerminalRequest(
                run.runId(),
                new StateStoreSlot("42", run.agentStateSessionId()),
                Set.of(AgentRunStatus.RUNNING),
                terminalStatus,
                outputType,
                errorCode,
                errorMessage,
                envelope);
    }

    private List<AgentModelCallUsage> usages(String runId) {
        return usageMapper.selectList(
                new LambdaQueryWrapper<AgentModelCallUsage>()
                        .eq(AgentModelCallUsage::getRunId, runId)
                        .orderByAsc(AgentModelCallUsage::getId));
    }

    private AgentModelCallUsage usage(String runId, String modelCallId) {
        AgentModelCallUsage row = usageMapper.selectOne(
                new LambdaQueryWrapper<AgentModelCallUsage>()
                        .eq(AgentModelCallUsage::getRunId, runId)
                        .eq(AgentModelCallUsage::getModelCallId, modelCallId));
        assertThat(row).isNotNull();
        return row;
    }

    private AgentModelCallUsage requireUsage(long usageId) {
        AgentModelCallUsage row = usageMapper.selectById(usageId);
        assertThat(row).isNotNull();
        return row;
    }

    private AgentRun requireRun(String runId) {
        AgentRun run = runMapper.selectOne(new LambdaQueryWrapper<AgentRun>()
                .eq(AgentRun::getRunId, runId));
        assertThat(run).isNotNull();
        return run;
    }

    private <T> T await(reactor.core.publisher.Mono<T> value) {
        // Blocking is restricted to this integration-test boundary.
        return value.block(Duration.ofSeconds(30));
    }

    private static String uniqueId(String prefix) {
        String boundedPrefix = prefix.length() <= 24 ? prefix : prefix.substring(0, 24);
        return boundedPrefix + '-' + UUID.randomUUID().toString().replace("-", "");
    }

    @FunctionalInterface
    private interface ThrowingBooleanSupplier {
        boolean getAsBoolean() throws Exception;
    }
}
