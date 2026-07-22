package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentTerminalOutputType;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.agentscope.context.ProjectContext;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelKey;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentCommand;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRun;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.AgentRunRedisSignalService;
import com.stonewu.fusion.service.ai.run.PlatformSubAgentRunService;
import com.stonewu.fusion.service.ai.run.RunExecutionSupervisor;
import com.stonewu.fusion.service.ai.run.RunTerminalCoordinator;
import com.stonewu.fusion.service.ai.run.RunShutdownCancellationPort;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.ChildRunIdentityConflictException;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.StartAgentExecutionCommand;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.UserMessage;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Flux;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;

import java.time.Duration;
import java.time.Instant;
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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@Testcontainers(disabledWithoutDocker = false)
class PlatformSubAgentRunServiceIT {

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
        properties.add("fusion.agentscope.v2.execution.instance-id", () -> "child-node");
    }

    @Autowired
    private PlatformSubAgentRunService service;

    @Autowired
    private AgentRunCoordinator coordinator;

    @Autowired
    private AgentConversationService conversationService;

    @Autowired
    private AgentRunMapper runMapper;

    @Autowired
    private RunShutdownCancellationPort shutdownCancellation;

    @Autowired
    private RunTerminalCoordinator terminalCoordinator;

    @MockitoBean
    private RunExecutionSupervisor supervisor;

    @MockitoBean
    private AgentRunRedisSignalService signals;

    @BeforeEach
    void configureMocks() {
        when(supervisor.start(any(StartAgentExecutionCommand.class))).thenReturn(Mono.empty());
        when(supervisor.interruptOwned(anyString(), anyString(), anyLong(), any()))
                .thenReturn(Mono.just(true));
        when(supervisor.shutdown(any())).thenReturn(Mono.empty());
        when(signals.publishCancel(anyString())).thenReturn(Mono.empty());
        when(signals.wakeupsWhenSubscribed(anyString()))
                .thenReturn(Mono.just(Flux.never()));
    }

    @Test
    void persistsAndStartsOneIdempotentChildWithoutExposingNativeTaskId() {
        StartedAgentRun parent = startParent();
        PlatformSubAgentCommand command = childCommand(parent, "tool-1", "researcher");

        PlatformSubAgentRun first = await(service.start(command));
        PlatformSubAgentRun retry = await(service.start(command));

        assertThat(retry).isEqualTo(first);
        assertThat(first.status()).isEqualTo(AgentRunStatus.RUNNING);
        assertThat(first.childRunId()).isNotBlank().isNotEqualTo(parent.runId());
        AgentRun child = run(first.childRunId());
        assertThat(child.getParentRunId()).isEqualTo(parent.runId());
        assertThat(child.getParentToolCallId()).isEqualTo("tool-1");
        assertThat(child.getAgentName()).isEqualTo("researcher");
        assertThat(child.getAgentStateSessionId())
                .isNotEqualTo(parent.agentStateSessionId())
                .startsWith("afv-child:");
        assertThat(child.getDeadlineAt().toInstant(java.time.ZoneOffset.UTC))
                .isEqualTo(command.deadline().truncatedTo(ChronoUnit.MILLIS));
        ArgumentCaptor<StartAgentExecutionCommand> execution =
                ArgumentCaptor.forClass(StartAgentExecutionCommand.class);
        verify(supervisor, times(1)).start(execution.capture());
        assertThat(execution.getValue().runtimeContextRequest()
                .conversation().agentStateSessionId())
                .isEqualTo(child.getAgentStateSessionId());

        PlatformSubAgentCommand changedIdentity = childCommand(
                parent, "tool-1", "different-agent");
        assertThatThrownBy(() -> await(service.start(changedIdentity)))
                .isInstanceOf(ChildRunIdentityConflictException.class);
    }

    @Test
    void rejectsStaleParentOwnershipAndCancelsOnlyDescendants() {
        StartedAgentRun parent = startParent();
        PlatformSubAgentCommand stale = new PlatformSubAgentCommand(
                parent.runId(),
                "tool-stale",
                parent.ownerInstanceId(),
                parent.ownerEpoch() + 1,
                "researcher",
                childSpec(),
                messages(),
                new ProjectContext(77L),
                parent.deadline().minusSeconds(1));
        assertThatThrownBy(() -> await(service.start(stale)))
                .isInstanceOf(BusinessException.class)
                .satisfies(failure -> assertThat(((BusinessException) failure).getCode())
                        .isEqualTo(409));

        PlatformSubAgentRun child = await(service.start(
                childCommand(parent, "tool-cancel", "researcher")));
        await(service.cancelChildren(parent.runId()));

        assertThat(run(child.childRunId()).getStatus())
                .isEqualTo(AgentRunStatus.CANCEL_REQUESTED.name());
        assertThat(run(parent.runId()).getStatus())
                .isEqualTo(AgentRunStatus.RUNNING.name());
    }

    @Test
    void awaitsTheDurableChildTerminalInsteadOfReturningRunningAsSuccess() {
        StartedAgentRun parent = startParent();
        PlatformSubAgentRun started = await(service.start(
                childCommand(parent, "tool-await", "researcher")));
        AgentRun child = run(started.childRunId());
        assertThat(await(terminalCoordinator.terminateOwned(
                new RunTerminalRequest(
                        child.getRunId(),
                        new StateStoreSlot(
                                String.valueOf(child.getUserId()),
                                child.getAgentStateSessionId()),
                        Set.of(AgentRunStatus.RUNNING),
                        AgentRunStatus.COMPLETED,
                        AgentTerminalOutputType.DONE,
                        null,
                        null,
                        new AgentEventEnvelope(
                                unique("child-terminal"),
                                "RUN_TERMINAL",
                                "main/researcher",
                                null,
                                null,
                                null,
                                "tool-await",
                                "researcher",
                                "DONE",
                                JsonNodeFactory.instance.objectNode()
                                        .put("outputType", "DONE")
                                        .put("finished", true),
                                Instant.now())),
                child.getOwnerInstanceId(),
                child.getOwnerEpoch()))).isPresent();

        PlatformSubAgentRun completed = await(service.awaitCompletion(started));

        assertThat(completed.status()).isEqualTo(AgentRunStatus.COMPLETED);
        assertThat(completed.childRunId()).isEqualTo(started.childRunId());
        assertThat(completed.parentToolCallId()).isEqualTo("tool-await");
        assertThat(completed.error()).isNull();
    }

    @Test
    void serializesParentCancellationAgainstConcurrentChildAdmission() throws Exception {
        StartedAgentRun parent = startParent();
        PlatformSubAgentCommand command = childCommand(
                parent, "tool-race", "researcher");
        CountDownLatch start = new CountDownLatch(1);

        try (ExecutorService pool = Executors.newFixedThreadPool(2)) {
            Future<?> childStart = pool.submit(() -> {
                await(start);
                try {
                    await(service.start(command));
                } catch (RuntimeException expectedWhenCancellationWins) {
                    // Either serialization order is valid; durable state is asserted below.
                }
            });
            Future<?> cancellation = pool.submit(() -> {
                await(start);
                await(shutdownCancellation.request(parent.runId()));
            });
            start.countDown();
            childStart.get(15, TimeUnit.SECONDS);
            cancellation.get(15, TimeUnit.SECONDS);
        }

        assertThat(run(parent.runId()).getStatus())
                .isEqualTo(AgentRunStatus.CANCEL_REQUESTED.name());
        assertThat(runMapper.selectActiveChildren(parent.runId()))
                .allSatisfy(child -> assertThat(child.getStatus())
                        .isEqualTo(AgentRunStatus.CANCEL_REQUESTED.name()));
    }

    private StartedAgentRun startParent() {
        long userId = 42L;
        long projectId = 77L;
        String conversationId = unique("conversation");
        conversationService.createOrUpdate(
                conversationId, userId, projectId, "project", projectId,
                "assistant", "child test", "chat");
        AgentKernelSnapshot snapshot = new CanonicalAgentKernelSnapshotBuilder()
                .build(new AgentKernelSnapshotPayload(
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
        Instant deadline = Instant.now().plusSeconds(60)
                .truncatedTo(ChronoUnit.MILLIS);
        return await(coordinator.start(new StartAgentRunCommand(
                unique("parent"),
                conversationId,
                userId,
                projectId,
                "assistant",
                null,
                null,
                null,
                "parent-session-" + unique("session"),
                snapshot,
                "parent-node",
                Duration.ofSeconds(30),
                deadline,
                "parent input",
                null)));
    }

    private PlatformSubAgentCommand childCommand(
            StartedAgentRun parent, String toolCallId, String agentName) {
        return new PlatformSubAgentCommand(
                parent.runId(),
                toolCallId,
                parent.ownerInstanceId(),
                parent.ownerEpoch(),
                agentName,
                childSpec(),
                messages(),
                new ProjectContext(77L),
                parent.deadline().minusSeconds(1));
    }

    private AgentKernelSpec childSpec() {
        AiModel model = AiModel.builder()
                .id(9L)
                .name("test")
                .code("test-model")
                .modelFamily("openai")
                .modelProtocol("openai")
                .modelType(1)
                .status(1)
                .config("{}")
                .build();
        AgentKernelKey key = AgentKernelKey.create(
                "researcher",
                "b".repeat(64),
                AgentKernelKey.promptVersion("child system"),
                List.of(),
                "v1");
        return new AgentKernelSpec(
                key,
                model,
                "researcher",
                "researcher",
                "child",
                "child system",
                5,
                List.of(),
                Set.of(),
                "v1");
    }

    private List<Msg> messages() {
        return List.of(new UserMessage(List.of(
                TextBlock.builder().text("research this").build())));
    }

    private AgentRun run(String runId) {
        return runMapper.selectOne(new LambdaQueryWrapper<AgentRun>()
                .eq(AgentRun::getRunId, runId));
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
