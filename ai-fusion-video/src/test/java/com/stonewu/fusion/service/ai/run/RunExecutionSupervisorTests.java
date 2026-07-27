package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentRuntimeErrorCode;
import com.stonewu.fusion.service.ai.agentscope.context.AgentConversationContext;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextRequest;
import com.stonewu.fusion.service.ai.agentscope.context.AuthenticatedUserContext;
import com.stonewu.fusion.service.ai.agentscope.context.CancellationContext;
import com.stonewu.fusion.service.ai.agentscope.context.PipelineRequestContext;
import com.stonewu.fusion.service.ai.agentscope.context.ParentAgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelKey;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.kernel.HarnessLeaseCache;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.kernel.RunConfigUnavailableException;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import com.stonewu.fusion.service.ai.run.model.ExecutionStopReason;
import com.stonewu.fusion.service.ai.run.model.ResumeAgentExecutionCommand;
import com.stonewu.fusion.service.ai.run.model.ResumedAgentRun;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.StartAgentExecutionCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.UserMessage;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import reactor.test.StepVerifier;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RunExecutionSupervisorTests {

    private final ObjectMapper objectMapper = new ObjectMapper().findAndRegisterModules();

    @Test
    void startReturnsOnlyAfterServerOwnedSubscriptionIsLaunched() {
        Fixture fixture = fixture();
        AtomicBoolean subscribed = new AtomicBoolean();
        AtomicInteger closes = new AtomicInteger();
        AtomicReference<ExecutionStopReason> interrupted = new AtomicReference<>();
        AgentExecution execution = execution(
                fixture.command.run(),
                Flux.<AgentEventEnvelope>never()
                        .doOnSubscribe(ignored -> subscribed.set(true)),
                interrupted,
                closes);
        when(fixture.executionFactory.start(
                anyString(), anyString(), anyLong(), anyString(),
                any(), any(), any(), any())).thenReturn(Mono.just(execution));

        StepVerifier.create(fixture.supervisor.start(fixture.command)).verifyComplete();

        assertThat(subscribed).isTrue();
        assertThat(fixture.registry.size()).isEqualTo(1);
        verify(fixture.executionFactory).start(
                fixture.command.run().runId(),
                fixture.command.run().ownerInstanceId(),
                fixture.command.run().ownerEpoch(),
                fixture.command.run().agentStateSessionId(),
                fixture.command.messages(),
                fixture.command.kernelSpec(),
                fixture.command.runtimeContextRequest(),
                fixture.command.run().deadline());

        StepVerifier.create(fixture.supervisor.interruptOwned(
                        fixture.command.run().runId(),
                        fixture.command.run().ownerInstanceId(),
                        fixture.command.run().ownerEpoch() + 1,
                        ExecutionStopReason.CANCEL_REQUESTED))
                .expectNext(false)
                .verifyComplete();
        StepVerifier.create(fixture.supervisor.interruptOwned(
                        fixture.command.run().runId(),
                        fixture.command.run().ownerInstanceId(),
                        fixture.command.run().ownerEpoch(),
                        ExecutionStopReason.CANCEL_REQUESTED))
                .expectNext(true)
                .verifyComplete();
        StepVerifier.create(fixture.registry.awaitEmpty(Duration.ofSeconds(1)))
                .verifyComplete();
        assertThat(interrupted.get()).isEqualTo(ExecutionStopReason.CANCEL_REQUESTED);
        assertThat(closes).hasValue(1);
    }

    @Test
    void journalsEventsAndCompletesRunExactlyOnce() {
        Fixture fixture = fixture();
        AgentEventEnvelope event = event("event-1");
        CommittedAgentEvent committed = mock(CommittedAgentEvent.class);
        when(fixture.journal.appendOwned(anyString(), anyString(), anyLong(), any()))
                .thenReturn(Mono.just(Optional.of(committed)));
        AtomicInteger closes = new AtomicInteger();
        when(fixture.executionFactory.start(
                anyString(), anyString(), anyLong(), anyString(),
                any(), any(), any(), any())).thenReturn(Mono.just(execution(
                        fixture.command.run(), Flux.just(event),
                        new AtomicReference<>(), closes)));

        StepVerifier.create(fixture.supervisor.start(fixture.command)).verifyComplete();
        StepVerifier.create(fixture.registry.awaitEmpty(Duration.ofSeconds(1)))
                .verifyComplete();

        ArgumentCaptor<RunTerminalRequest> terminal =
                ArgumentCaptor.forClass(RunTerminalRequest.class);
        verify(fixture.terminals).terminateOwned(
                terminal.capture(),
                eq(fixture.command.run().ownerInstanceId()),
                eq(fixture.command.run().ownerEpoch()));
        assertThat(terminal.getValue().terminalStatus().name()).isEqualTo("COMPLETED");
        assertThat(closes).hasValue(1);
    }

    @Test
    void durablyMirrorsCommittedChildEventsToTheOwnedParentRun() {
        Fixture fixture = fixture();
        ParentAgentRunContext parent = new ParentAgentRunContext(
                "parent-run-1",
                "parent-node-a",
                3,
                "parent-tool-1",
                "episode_scene_writer");
        AtomicReference<AgentEventEnvelope> childEnvelope = new AtomicReference<>();
        AtomicReference<AgentEventEnvelope> parentEnvelope = new AtomicReference<>();
        when(fixture.journal.appendOwned(anyString(), anyString(), anyLong(), any()))
                .thenAnswer(invocation -> {
                    String runId = invocation.getArgument(0, String.class);
                    AgentEventEnvelope envelope = invocation.getArgument(
                            3, AgentEventEnvelope.class);
                    if (fixture.command.run().runId().equals(runId)) {
                        childEnvelope.set(envelope);
                        return Mono.just(Optional.of(new CommittedAgentEvent(
                                11,
                                runId,
                                4,
                                envelope,
                                Instant.now())));
                    }
                    parentEnvelope.set(envelope);
                    return Mono.just(Optional.of(new CommittedAgentEvent(
                            12,
                            runId,
                            9,
                            envelope,
                            Instant.now())));
                });
        AgentEventEnvelope content = new AgentEventEnvelope(
                "child-content-1",
                "TEXT_BLOCK_DELTA",
                "main",
                "reply-child",
                "block-child",
                null,
                "parent-tool-1",
                "episode_scene_writer",
                "CONTENT",
                JsonNodeFactory.instance.objectNode().put("delta", "实时内容"),
                Instant.now());
        when(fixture.executionFactory.start(
                anyString(), anyString(), anyLong(), anyString(),
                any(), any(), any(), any())).thenReturn(Mono.just(
                        new AgentExecution(
                                fixture.command.run().runId(),
                                fixture.command.run().ownerInstanceId(),
                                fixture.command.run().ownerEpoch(),
                                7,
                                fixture.command.run().agentStateSessionId(),
                                parent,
                                Flux.just(content),
                                ignored -> Mono.empty(),
                                () -> { })));

        StepVerifier.create(fixture.supervisor.start(fixture.command)).verifyComplete();
        StepVerifier.create(fixture.registry.awaitEmpty(Duration.ofSeconds(1)))
                .verifyComplete();

        verify(fixture.journal).appendOwned(
                eq("parent-run-1"), eq("parent-node-a"), eq(3L), any());
        AgentEventEnvelope committedChild = childEnvelope.get();
        AgentEventEnvelope mirrored = parentEnvelope.get();
        assertThat(committedChild).isNotNull();
        assertThat(mirrored).isNotNull();
        assertThat(mirrored.rawEventId()).matches("child-[0-9a-f]{32}");
        assertThat(mirrored.rawEventType()).isEqualTo(committedChild.rawEventType());
        assertThat(mirrored.outputType()).isEqualTo(committedChild.outputType());
        assertThat(mirrored.parentToolCallId()).isEqualTo("parent-tool-1");
        assertThat(mirrored.agentName()).isEqualTo("episode_scene_writer");
        assertThat(mirrored.source()).isEqualTo("main/episode_scene_writer");
        assertThat(mirrored.payload().path("_platformMirroredChildEvent").asBoolean())
                .isTrue();
        assertThat(mirrored.payload().path("childRunId").asText())
                .isEqualTo(fixture.command.run().runId());
        assertThat(mirrored.payload().path("childSequence").asLong()).isEqualTo(4);
        assertThat(mirrored.payload().path("delta").asText()).isEqualTo("实时内容");
    }

    @Test
    void parentJournalOwnershipLossFailsTheChildExecutionClosed() {
        Fixture fixture = fixture();
        ParentAgentRunContext parent = new ParentAgentRunContext(
                "parent-run-1",
                "parent-node-a",
                3,
                "parent-tool-1",
                "episode_scene_writer");
        when(fixture.journal.appendOwned(anyString(), anyString(), anyLong(), any()))
                .thenAnswer(invocation -> {
                    String runId = invocation.getArgument(0, String.class);
                    AgentEventEnvelope envelope = invocation.getArgument(
                            3, AgentEventEnvelope.class);
                    if (!fixture.command.run().runId().equals(runId)) {
                        return Mono.just(Optional.empty());
                    }
                    return Mono.just(Optional.of(new CommittedAgentEvent(
                            11,
                            runId,
                            4,
                            envelope,
                            Instant.now())));
                });
        when(fixture.executionFactory.start(
                anyString(), anyString(), anyLong(), anyString(),
                any(), any(), any(), any())).thenReturn(Mono.just(
                        new AgentExecution(
                                fixture.command.run().runId(),
                                fixture.command.run().ownerInstanceId(),
                                fixture.command.run().ownerEpoch(),
                                7,
                                fixture.command.run().agentStateSessionId(),
                                parent,
                                Flux.just(event("child-event-1")),
                                ignored -> Mono.empty(),
                                () -> { })));

        StepVerifier.create(fixture.supervisor.start(fixture.command)).verifyComplete();
        StepVerifier.create(fixture.registry.awaitEmpty(Duration.ofSeconds(1)))
                .verifyComplete();

        ArgumentCaptor<RunTerminalRequest> terminal =
                ArgumentCaptor.forClass(RunTerminalRequest.class);
        verify(fixture.terminals).terminateOwned(
                terminal.capture(),
                eq(fixture.command.run().ownerInstanceId()),
                eq(fixture.command.run().ownerEpoch()));
        assertThat(terminal.getValue().terminalStatus()).isEqualTo(AgentRunStatus.FAILED);
        assertThat(terminal.getValue().errorCode())
                .isEqualTo(AgentRuntimeErrorCode.EVENT_PERSIST_FAILED);
        assertThat(terminal.getValue().errorMessage())
                .contains("Parent Agent run ownership was lost");
    }

    @Test
    void childCompletionKeepsDurableIdentityAndProjectsTheTerminalJournal() {
        Fixture fixture = fixture();
        CommittedAgentEvent terminalEvent = mock(CommittedAgentEvent.class);
        when(terminalEvent.runId()).thenReturn(fixture.command.run().runId());
        when(terminalEvent.sequence()).thenReturn(7L);
        when(fixture.terminals.terminateOwned(any(), anyString(), anyLong()))
                .thenReturn(Mono.just(Optional.of(terminalEvent)));
        when(fixture.executionFactory.start(
                anyString(), anyString(), anyLong(), anyString(),
                any(), any(), any(), any())).thenReturn(Mono.just(
                        new AgentExecution(
                                fixture.command.run().runId(),
                                fixture.command.run().ownerInstanceId(),
                                fixture.command.run().ownerEpoch(),
                                7,
                                fixture.command.run().agentStateSessionId(),
                                "parent-tool-1",
                                "episode_scene_writer",
                                Flux.empty(),
                                ignored -> Mono.empty(),
                                () -> { })));

        StepVerifier.create(fixture.supervisor.start(fixture.command)).verifyComplete();
        StepVerifier.create(fixture.registry.awaitEmpty(Duration.ofSeconds(1)))
                .verifyComplete();

        ArgumentCaptor<RunTerminalRequest> terminal =
                ArgumentCaptor.forClass(RunTerminalRequest.class);
        verify(fixture.terminals).terminateOwned(
                terminal.capture(),
                eq(fixture.command.run().ownerInstanceId()),
                eq(fixture.command.run().ownerEpoch()));
        assertThat(terminal.getValue().terminalEnvelope().parentToolCallId())
                .isEqualTo("parent-tool-1");
        assertThat(terminal.getValue().terminalEnvelope().agentName())
                .isEqualTo("episode_scene_writer");
        assertThat(terminal.getValue().terminalEnvelope().source())
                .isEqualTo("main/episode_scene_writer");
        verify(fixture.projections).projectThrough(
                fixture.command.run().runId(), 7L);
    }

    @Test
    void deadlineInterruptsTheExactOwnedExecution() {
        Fixture fixture = fixture(Duration.ofMillis(500));
        AtomicReference<ExecutionStopReason> interrupted = new AtomicReference<>();
        when(fixture.executionFactory.start(
                anyString(), anyString(), anyLong(), anyString(),
                any(), any(), any(), any())).thenReturn(Mono.just(execution(
                        fixture.command.run(), Flux.<AgentEventEnvelope>never(), interrupted,
                        new AtomicInteger())));

        StepVerifier.create(fixture.supervisor.start(fixture.command)).verifyComplete();
        StepVerifier.create(fixture.registry.awaitEmpty(Duration.ofSeconds(2)))
                .verifyComplete();

        assertThat(interrupted.get()).isEqualTo(ExecutionStopReason.DEADLINE);
    }

    @Test
    void resumeFailsClosedWhenPersistedKernelCannotBeResolved() {
        Fixture fixture = fixture();
        StartedAgentRun started = fixture.command.run();
        ResumedAgentRun resumed = new ResumedAgentRun(
                started.runId(),
                started.conversationId(),
                started.agentStateSessionId(),
                started.kernelSnapshot().fingerprint(),
                started.kernelSnapshot().snapshotJson(),
                1,
                started.ownerInstanceId(),
                2,
                started.leaseUntil(),
                started.deadline());
        AgentScopeRuntimeContextRequest runtime = runtime(
                started.runId(), started.ownerInstanceId(), 2, started.deadline());
        when(fixture.executionFactory.resolve(started.kernelSnapshot()))
                .thenReturn(Mono.error(new RunConfigUnavailableException("missing model")));

        StepVerifier.create(fixture.supervisor.resume(new ResumeAgentExecutionCommand(
                        resumed,
                        fixture.command.messages(),
                        started.kernelSnapshot(),
                        runtime)))
                .verifyComplete();

        ArgumentCaptor<RunTerminalRequest> terminal =
                ArgumentCaptor.forClass(RunTerminalRequest.class);
        verify(fixture.terminals).terminateOwned(
                terminal.capture(), eq(started.ownerInstanceId()), eq(2L));
        assertThat(terminal.getValue().errorCode())
                .isEqualTo(AgentRuntimeErrorCode.RUN_CONFIG_UNAVAILABLE);
    }

    @Test
    void overflowInterruptsProducerAndWritesBackpressureFailure() {
        Fixture fixture = fixture(Duration.ofSeconds(30), 2);
        CommittedAgentEvent committed = mock(CommittedAgentEvent.class);
        when(fixture.journal.appendOwned(anyString(), anyString(), anyLong(), any()))
                .thenAnswer(ignored -> Mono.delay(Duration.ofMillis(100))
                        .map(tick -> Optional.of(committed)));
        when(fixture.executionFactory.start(
                anyString(), anyString(), anyLong(), anyString(),
                any(), any(), any(), any())).thenReturn(Mono.just(execution(
                        fixture.command.run(),
                        Flux.just(event("event-1"), event("event-2"),
                                event("event-3"), event("event-4"),
                                event("event-5"), event("event-6"),
                                event("event-7"), event("event-8")),
                        new AtomicReference<>(),
                        new AtomicInteger())));

        StepVerifier.create(fixture.supervisor.start(fixture.command)).verifyComplete();
        StepVerifier.create(fixture.registry.awaitEmpty(Duration.ofSeconds(2)))
                .verifyComplete();

        ArgumentCaptor<RunTerminalRequest> terminal =
                ArgumentCaptor.forClass(RunTerminalRequest.class);
        verify(fixture.terminals).terminateOwned(
                terminal.capture(),
                eq(fixture.command.run().ownerInstanceId()),
                eq(fixture.command.run().ownerEpoch()));
        assertThat(terminal.getValue().errorCode())
                .isEqualTo(AgentRuntimeErrorCode.AGENT_EVENT_BACKPRESSURE_OVERFLOW);
    }

    @Test
    void shutdownRejectsNewStartsAndPersistsCancellationForSurvivors() {
        Fixture fixture = fixture(Duration.ofSeconds(30));
        AtomicReference<ExecutionStopReason> interrupted = new AtomicReference<>();
        when(fixture.executionFactory.start(
                anyString(), anyString(), anyLong(), anyString(),
                any(), any(), any(), any())).thenReturn(Mono.just(execution(
                        fixture.command.run(),
                        Flux.<AgentEventEnvelope>never(),
                        interrupted,
                        new AtomicInteger())));

        StepVerifier.create(fixture.supervisor.start(fixture.command)).verifyComplete();
        StepVerifier.create(fixture.supervisor.shutdown(Duration.ofMillis(20)))
                .verifyComplete();

        verify(fixture.shutdownCancellation).request(fixture.command.run().runId());
        verify(fixture.cache).drainAndClose(Duration.ofMillis(20));
        assertThat(interrupted.get()).isEqualTo(ExecutionStopReason.SHUTDOWN);
        StepVerifier.create(fixture.supervisor.start(fixture.command))
                .expectErrorMessage("Agent runtime is shutting down")
                .verify();
    }

    private Fixture fixture() {
        return fixture(Duration.ofSeconds(30));
    }

    private Fixture fixture(Duration deadlineDelay) {
        return fixture(deadlineDelay, 32);
    }

    private Fixture fixture(Duration deadlineDelay, int maxEvents) {
        AgentExecutionFactory factory = mock(AgentExecutionFactory.class);
        AgentEventJournal journal = mock(AgentEventJournal.class);
        RunTerminalCoordinator terminals = mock(RunTerminalCoordinator.class);
        when(terminals.terminateOwned(any(), anyString(), anyLong()))
                .thenReturn(Mono.just(Optional.empty()));
        HarnessLeaseCache cache = mock(HarnessLeaseCache.class);
        AgentMessageProjectionService projections = mock(
                AgentMessageProjectionService.class);
        when(projections.projectThrough(anyString(), anyLong()))
                .thenReturn(Mono.empty());
        when(cache.drainAndClose(any())).thenReturn(Mono.empty());
        RunShutdownCancellationPort shutdown = mock(RunShutdownCancellationPort.class);
        when(shutdown.request(anyString())).thenReturn(Mono.empty());
        RunLeaseGuard leases = mock(RunLeaseGuard.class);
        AgentRunRedisSignalService signals = mock(AgentRunRedisSignalService.class);
        when(signals.cancellations(anyString())).thenReturn(Flux.never());
        when(signals.publishWakeup(anyString(), anyLong())).thenReturn(Mono.empty());
        OwnedExecutionRegistry registry = new OwnedExecutionRegistry(
                objectMapper, maxEvents, 1024 * 1024,
                Schedulers.parallel(), Clock.systemUTC());
        AgentEventChunkCoalescer coalescer = new AgentEventChunkCoalescer(
                objectMapper,
                new AgentEventChunkCoalescer.ChunkPolicy(Duration.ofMillis(5), 1024),
                Schedulers.parallel(),
                32);
        DefaultRunExecutionSupervisor supervisor = new DefaultRunExecutionSupervisor(
                factory,
                registry,
                coalescer,
                journal,
                terminals,
                projections,
                cache,
                new CanonicalAgentKernelSnapshotBuilder(objectMapper),
                shutdown,
                leases,
                signals,
                new AgentEventEnvelopeSanitizer());
        Instant deadline = Instant.now().plus(deadlineDelay);
        AgentKernelSpec spec = spec();
        AgentKernelSnapshot snapshot =
                new CanonicalAgentKernelSnapshotBuilder(objectMapper).build(spec);
        StartedAgentRun started = new StartedAgentRun(
                "run-1",
                "conversation-1",
                "session-1",
                "node-a",
                1,
                deadline.minusMillis(1),
                deadline,
                snapshot,
                1);
        List<Msg> messages = List.of(new UserMessage(List.of(
                TextBlock.builder().text("hello").build())));
        StartAgentExecutionCommand command = new StartAgentExecutionCommand(
                started,
                messages,
                snapshot,
                spec,
                runtime(started.runId(), started.ownerInstanceId(), 1, deadline));
        return new Fixture(
                factory, registry, journal, terminals, cache, shutdown,
                projections, supervisor, command);
    }

    private AgentExecution execution(
            StartedAgentRun run,
            Flux<AgentEventEnvelope> events,
            AtomicReference<ExecutionStopReason> interrupted,
            AtomicInteger closes) {
        return new AgentExecution(
                run.runId(),
                run.ownerInstanceId(),
                run.ownerEpoch(),
                7,
                run.agentStateSessionId(),
                events,
                reason -> Mono.fromRunnable(() -> interrupted.set(reason)),
                closes::incrementAndGet);
    }

    private AgentKernelSpec spec() {
        AiModel model = AiModel.builder()
                .id(1L)
                .name("test")
                .code("test-model")
                .modelProtocol("openai")
                .modelType(1)
                .status(1)
                .config("{}")
                .build();
        String fingerprint = "a".repeat(64);
        AgentKernelKey key = AgentKernelKey.create(
                "assistant", fingerprint,
                AgentKernelKey.promptVersion("system"),
                List.of(), "v1");
        return new AgentKernelSpec(
                key,
                model,
                "assistant",
                "assistant",
                "test assistant",
                "system",
                5,
                List.of(),
                Set.of(),
                "v1");
    }

    private AgentScopeRuntimeContextRequest runtime(
            String runId, String owner, long epoch, Instant deadline) {
        return new AgentScopeRuntimeContextRequest(
                new AuthenticatedUserContext(7L),
                new AgentConversationContext(
                        "conversation-1", "assistant", "session-1"),
                new AgentRunContext(runId, owner, epoch, deadline),
                null,
                new PipelineRequestContext(runId, PipelineRequestContext.Kind.CHAT),
                null,
                CancellationContext.noop());
    }

    private AgentEventEnvelope event(String id) {
        return new AgentEventEnvelope(
                id,
                "TEXT_BLOCK_END",
                "main",
                "reply",
                "block",
                null,
                null,
                null,
                null,
                JsonNodeFactory.instance.objectNode(),
                Instant.now());
    }

    private record Fixture(
            AgentExecutionFactory executionFactory,
            OwnedExecutionRegistry registry,
            AgentEventJournal journal,
            RunTerminalCoordinator terminals,
            HarnessLeaseCache cache,
            RunShutdownCancellationPort shutdownCancellation,
            AgentMessageProjectionService projections,
            DefaultRunExecutionSupervisor supervisor,
            StartAgentExecutionCommand command) {
    }
}
