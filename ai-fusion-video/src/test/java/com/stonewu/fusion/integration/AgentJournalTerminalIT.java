package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentRuntimeErrorCode;
import com.stonewu.fusion.enums.ai.AgentTerminalOutputType;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreFailureGuard;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import com.stonewu.fusion.service.ai.run.AgentEventJournal;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.RunTerminalCoordinator;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import com.stonewu.fusion.service.ai.run.model.SystemTerminalActor;
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
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.LongStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@Testcontainers(disabledWithoutDocker = false)
class AgentJournalTerminalIT {

    private static final int CONCURRENT_APPENDS = 32;
    private static final int TERMINAL_RACE_REPETITIONS = 12;

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
        properties.add("spring.datasource.hikari.maximum-pool-size", () -> 40);
        properties.add("spring.datasource.hikari.minimum-idle", () -> 0);
        properties.add("app.agentscope.runtime.journal-threads", () -> 32);
    }

    @Autowired
    private AgentEventJournal journal;

    @Autowired
    private RunTerminalCoordinator terminalCoordinator;

    @Autowired
    private AgentRunCoordinator runCoordinator;

    @Autowired
    private AgentConversationService conversationService;

    @Autowired
    private AgentRunMapper runMapper;

    @Autowired
    private AgentEventMapper eventMapper;

    @Autowired
    private StateStoreFailureGuard stateStoreFailures;

    @Test
    void allocatesGapFreePerRunSequenceUnderConcurrentAppends() throws Exception {
        StartedAgentRun run = startRoot("append-race");
        CountDownLatch ready = new CountDownLatch(CONCURRENT_APPENDS);
        CountDownLatch start = new CountDownLatch(1);
        ExecutorService callers = Executors.newFixedThreadPool(CONCURRENT_APPENDS);
        List<Future<Optional<CommittedAgentEvent>>> futures = new ArrayList<>();
        try {
            for (int index = 0; index < CONCURRENT_APPENDS; index++) {
                int eventIndex = index;
                futures.add(callers.submit(() -> {
                    ready.countDown();
                    if (!start.await(20, TimeUnit.SECONDS)) {
                        throw new IllegalStateException("append callers did not start together");
                    }
                    return await(journal.appendOwned(
                            run.runId(), run.ownerInstanceId(), run.ownerEpoch(),
                            contentEvent("append-" + eventIndex)));
                }));
            }
            assertThat(ready.await(20, TimeUnit.SECONDS)).isTrue();
            start.countDown();

            List<Long> sequences = new ArrayList<>();
            for (Future<Optional<CommittedAgentEvent>> future : futures) {
                Optional<CommittedAgentEvent> committed = future.get(30, TimeUnit.SECONDS);
                assertThat(committed).isPresent();
                sequences.add(committed.orElseThrow().sequence());
            }
            sequences.sort(Comparator.naturalOrder());
            assertThat(sequences).containsExactlyElementsOf(
                    LongStream.rangeClosed(1, CONCURRENT_APPENDS).boxed().toList());
        } finally {
            start.countDown();
            callers.shutdownNow();
            assertThat(callers.awaitTermination(20, TimeUnit.SECONDS)).isTrue();
        }

        assertThat(events(run.runId()))
                .extracting(AgentEvent::getSequenceNo)
                .containsExactlyElementsOf(
                        LongStream.rangeClosed(1, CONCURRENT_APPENDS).boxed().toList());
        assertThat(requireRun(run.runId()).getNextSequence())
                .isEqualTo(CONCURRENT_APPENDS + 1L);
    }

    @Test
    void rejectsOwnerEpochLeaseAndStatusMismatchWithoutChangingRunOrEvents() {
        StartedAgentRun ownerRun = startRoot("owner-reject");
        assertRejectedAppend(ownerRun, "wrong-owner", ownerRun.ownerEpoch());
        assertRejectedAppend(ownerRun, ownerRun.ownerInstanceId(), ownerRun.ownerEpoch() + 1);

        StartedAgentRun equalLease = startRoot("equal-lease");
        LocalDateTime databaseNow = runMapper.selectDatabaseNow();
        updateRun(equalLease.runId(), AgentRun::getLeaseUntil, databaseNow);
        assertRejectedAppend(equalLease, equalLease.ownerInstanceId(), equalLease.ownerEpoch());

        StartedAgentRun expiredLease = startRoot("expired-lease");
        updateRun(expiredLease.runId(), AgentRun::getLeaseUntil,
                runMapper.selectDatabaseNow().minusSeconds(1));
        assertRejectedAppend(expiredLease,
                expiredLease.ownerInstanceId(), expiredLease.ownerEpoch());

        StartedAgentRun wrongStatus = startRoot("status-reject");
        updateRun(wrongStatus.runId(), AgentRun::getStatus,
                AgentRunStatus.WAITING_CONFIRMATION.name());
        assertRejectedAppend(wrongStatus,
                wrongStatus.ownerInstanceId(), wrongStatus.ownerEpoch());
    }

    @Test
    void commitsExactlyOneTerminalAcrossCompleteFailCancelRacesAndRejectsLaterEvents()
            throws Exception {
        ExecutorService callers = Executors.newFixedThreadPool(3);
        try {
            for (int repetition = 0; repetition < TERMINAL_RACE_REPETITIONS; repetition++) {
                StartedAgentRun run = startRoot("terminal-race-" + repetition);
                StateStoreSlot slot = slot(run);
                CountDownLatch ready = new CountDownLatch(3);
                CountDownLatch start = new CountDownLatch(1);
                List<RunTerminalRequest> requests = List.of(
                        completed(run, slot),
                        failed(run, slot, AgentRuntimeErrorCode.AGENTSCOPE_INTERNAL_ERROR),
                        cancelled(run, slot, Set.of(AgentRunStatus.RUNNING)));
                List<Future<Optional<CommittedAgentEvent>>> futures = new ArrayList<>();
                for (RunTerminalRequest request : requests) {
                    futures.add(callers.submit(() -> {
                        ready.countDown();
                        if (!start.await(15, TimeUnit.SECONDS)) {
                            throw new IllegalStateException("terminal callers did not start together");
                        }
                        return await(terminalCoordinator.terminateOwned(
                                request, run.ownerInstanceId(), run.ownerEpoch()));
                    }));
                }
                assertThat(ready.await(15, TimeUnit.SECONDS)).isTrue();
                start.countDown();

                List<CommittedAgentEvent> winners = new ArrayList<>();
                for (Future<Optional<CommittedAgentEvent>> future : futures) {
                    future.get(20, TimeUnit.SECONDS).ifPresent(winners::add);
                }
                assertThat(winners).singleElement()
                        .satisfies(winner -> assertThat(winner.sequence()).isEqualTo(1L));
                AgentRun persisted = requireRun(run.runId());
                assertThat(persisted.getTerminalSequence()).isEqualTo(1L);
                assertThat(persisted.getNextSequence()).isEqualTo(2L);
                assertThat(events(run.runId())).hasSize(1);
                assertThat(await(journal.appendOwned(
                        run.runId(), run.ownerInstanceId(), run.ownerEpoch(),
                        contentEvent("after-terminal"))))
                        .isEmpty();
                assertThat(events(run.runId())).hasSize(1);
            }
        } finally {
            callers.shutdownNow();
            assertThat(callers.awaitTermination(20, TimeUnit.SECONDS)).isTrue();
        }
    }

    @Test
    void enforcesSystemActorTransitionsAndFailsClosedOnStateStoreFailure() {
        StartedAgentRun cancellation = startRoot("system-cancellation");
        RunTerminalRequest runningCancel = cancelled(
                cancellation, slot(cancellation), Set.of(AgentRunStatus.RUNNING));
        assertThat(await(terminalCoordinator.terminateSystem(
                runningCancel, SystemTerminalActor.CANCELLATION_COORDINATOR)))
                .isEmpty();
        assertThat(events(cancellation.runId())).isEmpty();
        updateRun(cancellation.runId(), AgentRun::getStatus,
                AgentRunStatus.CANCEL_REQUESTED.name());
        assertThat(await(terminalCoordinator.terminateSystem(
                cancelled(cancellation, slot(cancellation),
                        Set.of(AgentRunStatus.CANCEL_REQUESTED)),
                SystemTerminalActor.CANCELLATION_COORDINATOR)))
                .isPresent();
        assertThat(requireRun(cancellation.runId()).getStatus())
                .isEqualTo(AgentRunStatus.CANCELLED.name());

        StartedAgentRun reconciled = startRoot("system-reconciler");
        RunTerminalRequest ownerLost = failed(
                reconciled, slot(reconciled), AgentRuntimeErrorCode.OWNER_LOST);
        assertThat(await(terminalCoordinator.terminateSystem(
                ownerLost, SystemTerminalActor.OWNER_RECONCILER)))
                .isEmpty();
        updateRun(reconciled.runId(), AgentRun::getLeaseUntil,
                runMapper.selectDatabaseNow().minusSeconds(1));
        assertThat(await(terminalCoordinator.terminateSystem(
                ownerLost, SystemTerminalActor.OWNER_RECONCILER)))
                .isPresent();
        AgentRun lost = requireRun(reconciled.runId());
        assertThat(lost.getStatus()).isEqualTo(AgentRunStatus.FAILED.name());
        assertThat(lost.getErrorCode()).isEqualTo(AgentRuntimeErrorCode.OWNER_LOST.name());

        StartedAgentRun cancelledByReconciler = startRoot("expired-cancel");
        updateRun(cancelledByReconciler.runId(), AgentRun::getStatus,
                AgentRunStatus.CANCEL_REQUESTED.name());
        updateRun(cancelledByReconciler.runId(), AgentRun::getLeaseUntil,
                runMapper.selectDatabaseNow().minusSeconds(1));
        assertThat(await(terminalCoordinator.terminateSystem(
                cancelled(cancelledByReconciler, slot(cancelledByReconciler),
                        Set.of(AgentRunStatus.CANCEL_REQUESTED)),
                SystemTerminalActor.OWNER_RECONCILER)))
                .isPresent();

        StartedAgentRun guarded = startRoot("state-store-failure");
        StateStoreSlot guardedSlot = slot(guarded);
        stateStoreFailures.record(
                guardedSlot, "save", new IllegalStateException("simulated state-store failure"));
        try {
            Optional<CommittedAgentEvent> committed = await(terminalCoordinator.terminateOwned(
                    completed(guarded, guardedSlot),
                    guarded.ownerInstanceId(), guarded.ownerEpoch()));
            assertThat(committed).isPresent();
            assertThat(committed.orElseThrow().outputType()).isEqualTo("ERROR");
            AgentRun failed = requireRun(guarded.runId());
            assertThat(failed.getStatus()).isEqualTo(AgentRunStatus.FAILED.name());
            assertThat(failed.getErrorCode())
                    .isEqualTo(AgentRuntimeErrorCode.STATE_STORE_FAILED.name());
            assertThat(events(guarded.runId()))
                    .singleElement()
                    .satisfies(event -> {
                        assertThat(event.getOutputType()).isEqualTo("ERROR");
                        assertThat(event.getRawEventType()).isEqualTo("STATE_STORE_FAILED");
                    });
        } finally {
            stateStoreFailures.clear(guardedSlot);
        }
    }

    @Test
    void rejectsForgedStateStoreSlotWithoutChangingRunOrEvents() {
        StartedAgentRun run = startRoot("forged-state-store-slot");
        StateStoreSlot authoritativeSlot = slot(run);
        StateStoreSlot forgedCleanSlot = new StateStoreSlot(
                "999", run.agentStateSessionId());
        stateStoreFailures.record(
                authoritativeSlot, "save",
                new IllegalStateException("simulated authoritative state-store failure"));
        try {
            AgentRun before = requireRun(run.runId());

            assertThatThrownBy(() -> await(terminalCoordinator.terminateOwned(
                    completed(run, forgedCleanSlot),
                    run.ownerInstanceId(), run.ownerEpoch())))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessage("Run terminal StateStore slot does not match persisted run identity");

            AgentRun after = requireRun(run.runId());
            assertThat(after.getStatus()).isEqualTo(AgentRunStatus.RUNNING.name());
            assertThat(after.getNextSequence()).isEqualTo(before.getNextSequence());
            assertThat(after.getTerminalSequence()).isNull();
            assertThat(events(run.runId())).isEmpty();
        } finally {
            stateStoreFailures.clear(authoritativeSlot);
        }
    }

    private void assertRejectedAppend(
            StartedAgentRun run, String ownerInstanceId, long ownerEpoch) {
        AgentRun before = requireRun(run.runId());
        int eventCount = events(run.runId()).size();
        assertThat(await(journal.appendOwned(
                run.runId(), ownerInstanceId, ownerEpoch,
                contentEvent("must-not-commit"))))
                .isEmpty();
        assertThat(await(terminalCoordinator.terminateOwned(
                completed(run, slot(run)), ownerInstanceId, ownerEpoch)))
                .isEmpty();
        AgentRun after = requireRun(run.runId());
        assertThat(after.getNextSequence()).isEqualTo(before.getNextSequence());
        assertThat(events(run.runId())).hasSize(eventCount);
    }

    private StartedAgentRun startRoot(String prefix) {
        long userId = 42L;
        long projectId = 7L;
        String conversationId = uniqueId("conversation-" + prefix);
        conversationService.createOrUpdate(
                conversationId, userId, projectId, "project", projectId,
                "assistant", "Journal terminal test", "chat");
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
                        "Journal test agent",
                        "You are a journal test agent.",
                        10,
                        "model-config-1",
                        1L,
                        "openai",
                        "gpt-test",
                        JsonNodeFactory.instance.objectNode().put("temperature", 0.2),
                        List.of(),
                        "0.6.3-test"));
    }

    private AgentEventEnvelope contentEvent(String id) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", "CONTENT")
                .put("content", id)
                .put("finished", false);
        return envelope(id, "TEXT_BLOCK_DELTA", "CONTENT", payload);
    }

    private RunTerminalRequest completed(StartedAgentRun run, StateStoreSlot slot) {
        ObjectNode payload = terminalPayload("DONE", null, null);
        return terminalRequest(
                run, slot, Set.of(AgentRunStatus.RUNNING),
                AgentRunStatus.COMPLETED, AgentTerminalOutputType.DONE,
                null, null, envelope(uniqueId("done"), "AGENT_END", "DONE", payload));
    }

    private RunTerminalRequest failed(
            StartedAgentRun run,
            StateStoreSlot slot,
            AgentRuntimeErrorCode errorCode) {
        String errorMessage = "terminal failure";
        ObjectNode payload = terminalPayload("ERROR", errorCode, errorMessage);
        return terminalRequest(
                run, slot, Set.of(AgentRunStatus.RUNNING),
                AgentRunStatus.FAILED, AgentTerminalOutputType.ERROR,
                errorCode, errorMessage,
                envelope(uniqueId("error"), "AGENT_ERROR", "ERROR", payload));
    }

    private RunTerminalRequest cancelled(
            StartedAgentRun run,
            StateStoreSlot slot,
            Set<AgentRunStatus> expectedStatuses) {
        String message = "run cancelled";
        ObjectNode payload = terminalPayload(
                "CANCELLED", AgentRuntimeErrorCode.RUN_CANCELLED, message);
        return terminalRequest(
                run, slot, expectedStatuses,
                AgentRunStatus.CANCELLED, AgentTerminalOutputType.CANCELLED,
                AgentRuntimeErrorCode.RUN_CANCELLED, message,
                envelope(uniqueId("cancel"), "REQUEST_STOP", "CANCELLED", payload));
    }

    private RunTerminalRequest terminalRequest(
            StartedAgentRun run,
            StateStoreSlot slot,
            Set<AgentRunStatus> expectedStatuses,
            AgentRunStatus status,
            AgentTerminalOutputType outputType,
            AgentRuntimeErrorCode errorCode,
            String errorMessage,
            AgentEventEnvelope envelope) {
        return new RunTerminalRequest(
                run.runId(), slot, expectedStatuses, status, outputType,
                errorCode, errorMessage, envelope);
    }

    private ObjectNode terminalPayload(
            String outputType,
            AgentRuntimeErrorCode errorCode,
            String errorMessage) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", outputType)
                .put("finished", true);
        if (errorCode != null) {
            payload.put("errorCode", errorCode.name());
        }
        if (errorMessage != null) {
            payload.put("error", errorMessage);
        }
        return payload;
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

    private StateStoreSlot slot(StartedAgentRun run) {
        return new StateStoreSlot("42", run.agentStateSessionId());
    }

    private AgentRun requireRun(String runId) {
        AgentRun run = runMapper.selectOne(new LambdaQueryWrapper<AgentRun>()
                .eq(AgentRun::getRunId, runId));
        assertThat(run).isNotNull();
        return run;
    }

    private List<AgentEvent> events(String runId) {
        return eventMapper.selectList(new LambdaQueryWrapper<AgentEvent>()
                .eq(AgentEvent::getRunId, runId)
                .orderByAsc(AgentEvent::getSequenceNo));
    }

    private <T> void updateRun(
            String runId,
            com.baomidou.mybatisplus.core.toolkit.support.SFunction<AgentRun, T> field,
            T value) {
        assertThat(runMapper.update(null, new LambdaUpdateWrapper<AgentRun>()
                .eq(AgentRun::getRunId, runId)
                .set(field, value))).isEqualTo(1);
    }

    private <T> T await(reactor.core.publisher.Mono<T> value) {
        // Blocking is restricted to this integration-test boundary.
        return value.block(Duration.ofSeconds(30));
    }

    private static String uniqueId(String prefix) {
        String boundedPrefix = prefix.length() <= 24 ? prefix : prefix.substring(0, 24);
        return boundedPrefix + '-' + UUID.randomUUID().toString().replace("-", "");
    }
}
