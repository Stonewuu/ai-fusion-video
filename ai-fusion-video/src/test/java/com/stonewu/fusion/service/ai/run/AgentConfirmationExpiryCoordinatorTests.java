<<<<<<< Updated upstream
package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.SystemTerminalActor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import reactor.test.StepVerifier;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentConfirmationExpiryCoordinatorTests {

    private AgentEventMapper eventMapper;
    private RunTerminalCoordinator terminals;
    private AgentMessageProjectionService projections;
    private AgentConfirmationExpiryCoordinator coordinator;

    @BeforeEach
    void setUp() {
        AgentRunMapper runMapper = mock(AgentRunMapper.class);
        eventMapper = mock(AgentEventMapper.class);
        terminals = mock(RunTerminalCoordinator.class);
        projections = mock(AgentMessageProjectionService.class);
        ObjectMapper objectMapper = new ObjectMapper().findAndRegisterModules();
        AgentRuntimeSchedulers schedulers = mock(AgentRuntimeSchedulers.class);
        when(schedulers.journal()).thenReturn(Schedulers.immediate());
        coordinator = new AgentConfirmationExpiryCoordinator(
                runMapper,
                eventMapper,
                objectMapper,
                terminals,
                projections,
                new AgentEventEnvelopeSanitizer(objectMapper),
                schedulers);
    }

    @Test
    void expiredApprovalTerminatesAsCancelledAndProjectsTheNotice() {
        AgentRun run = waitingRun(Instant.now().minusSeconds(1));
        String pendingToolCalls = """
                [{"toolCallId":"tool-1","toolName":"save_storyboard_scene_shots",\
                "argumentsPreview":"{}"}]
                """;
        AgentEvent candidate = AgentEvent.builder()
                .runId(run.getRunId())
                .rawEventType("REQUIRE_USER_CONFIRM")
                .payloadJson(JsonNodeFactory.instance.objectNode()
                        .put("pendingToolCallsJson", pendingToolCalls)
                        .toString())
                .build();
        when(eventMapper.selectConfirmationCandidate("run-1", "reply-1"))
                .thenReturn(candidate);
        when(terminals.terminateSystem(
                any(RunTerminalRequest.class),
                eq(SystemTerminalActor.CONFIRMATION_EXPIRER)))
                .thenAnswer(invocation -> {
                    RunTerminalRequest request = invocation.getArgument(0);
                    return Mono.just(Optional.of(new CommittedAgentEvent(
                            9,
                            run.getRunId(),
                            9,
                            request.terminalEnvelope(),
                            Instant.now())));
                });
        when(projections.projectThrough("run-1", 9)).thenReturn(Mono.empty());

        StepVerifier.create(coordinator.expireIfNeeded(run))
                .expectNext(true)
                .verifyComplete();

        ArgumentCaptor<RunTerminalRequest> request =
                ArgumentCaptor.forClass(RunTerminalRequest.class);
        verify(terminals).terminateSystem(
                request.capture(), eq(SystemTerminalActor.CONFIRMATION_EXPIRER));
        assertThat(request.getValue().terminalStatus())
                .isEqualTo(AgentRunStatus.CANCELLED);
        assertThat(request.getValue().terminalEnvelope().payload().path("content").asText())
                .isEqualTo(AgentConfirmationExpiryCoordinator.MESSAGE);
        assertThat(request.getValue().terminalEnvelope().payload()
                .path("pendingToolCalls").get(0).path("toolCallId").asText())
                .isEqualTo("tool-1");
        verify(projections).projectThrough("run-1", 9);
    }

    @Test
    void activeApprovalRemainsPending() {
        AgentRun run = waitingRun(Instant.now().plusSeconds(60));

        StepVerifier.create(coordinator.expireIfNeeded(run))
                .expectNext(false)
                .verifyComplete();

        verify(terminals, never()).terminateSystem(any(), any());
    }

    private AgentRun waitingRun(Instant expiresAt) {
        return AgentRun.builder()
                .runId("run-1")
                .conversationId("conversation-1")
                .userId(42L)
                .status(AgentRunStatus.WAITING_CONFIRMATION.name())
                .waitingReplyId("reply-1")
                .waitExpiresAt(LocalDateTime.ofInstant(expiresAt, ZoneOffset.UTC))
                .deadlineAt(LocalDateTime.ofInstant(expiresAt, ZoneOffset.UTC))
                .agentStateSessionId("session-1")
                .nextSequence(9L)
                .build();
    }
}
=======
package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.SystemTerminalActor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import reactor.test.StepVerifier;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentConfirmationExpiryCoordinatorTests {

    private AgentEventMapper eventMapper;
    private RunTerminalCoordinator terminals;
    private AgentMessageProjectionService projections;
    private AgentConfirmationExpiryCoordinator coordinator;

    @BeforeEach
    void setUp() {
        AgentRunMapper runMapper = mock(AgentRunMapper.class);
        eventMapper = mock(AgentEventMapper.class);
        terminals = mock(RunTerminalCoordinator.class);
        projections = mock(AgentMessageProjectionService.class);
        AgentRuntimeSchedulers schedulers = mock(AgentRuntimeSchedulers.class);
        when(schedulers.journal()).thenReturn(Schedulers.immediate());
        coordinator = new AgentConfirmationExpiryCoordinator(
                runMapper,
                eventMapper,
                new ObjectMapper().findAndRegisterModules(),
                terminals,
                projections,
                new AgentEventEnvelopeSanitizer(),
                schedulers);
    }

    @Test
    void expiredApprovalTerminatesAsCancelledAndProjectsTheNotice() {
        AgentRun run = waitingRun(Instant.now().minusSeconds(1));
        String pendingToolCalls = """
                [{"toolCallId":"tool-1","toolName":"save_storyboard_scene_shots",\
                "argumentsPreview":"{}"}]
                """;
        AgentEvent candidate = AgentEvent.builder()
                .runId(run.getRunId())
                .rawEventType("REQUIRE_USER_CONFIRM")
                .payloadJson(JsonNodeFactory.instance.objectNode()
                        .put("pendingToolCallsJson", pendingToolCalls)
                        .toString())
                .build();
        when(eventMapper.selectConfirmationCandidate("run-1", "reply-1"))
                .thenReturn(candidate);
        when(terminals.terminateSystem(
                any(RunTerminalRequest.class),
                eq(SystemTerminalActor.CONFIRMATION_EXPIRER)))
                .thenAnswer(invocation -> {
                    RunTerminalRequest request = invocation.getArgument(0);
                    return Mono.just(Optional.of(new CommittedAgentEvent(
                            9,
                            run.getRunId(),
                            9,
                            request.terminalEnvelope(),
                            Instant.now())));
                });
        when(projections.projectThrough("run-1", 9)).thenReturn(Mono.empty());

        StepVerifier.create(coordinator.expireIfNeeded(run))
                .expectNext(true)
                .verifyComplete();

        ArgumentCaptor<RunTerminalRequest> request =
                ArgumentCaptor.forClass(RunTerminalRequest.class);
        verify(terminals).terminateSystem(
                request.capture(), eq(SystemTerminalActor.CONFIRMATION_EXPIRER));
        assertThat(request.getValue().terminalStatus())
                .isEqualTo(AgentRunStatus.CANCELLED);
        assertThat(request.getValue().terminalEnvelope().payload().path("content").asText())
                .isEqualTo(AgentConfirmationExpiryCoordinator.MESSAGE);
        assertThat(request.getValue().terminalEnvelope().payload()
                .path("pendingToolCalls").get(0).path("toolCallId").asText())
                .isEqualTo("tool-1");
        verify(projections).projectThrough("run-1", 9);
    }

    @Test
    void activeApprovalRemainsPending() {
        AgentRun run = waitingRun(Instant.now().plusSeconds(60));

        StepVerifier.create(coordinator.expireIfNeeded(run))
                .expectNext(false)
                .verifyComplete();

        verify(terminals, never()).terminateSystem(any(), any());
    }

    private AgentRun waitingRun(Instant expiresAt) {
        return AgentRun.builder()
                .runId("run-1")
                .conversationId("conversation-1")
                .userId(42L)
                .status(AgentRunStatus.WAITING_CONFIRMATION.name())
                .waitingReplyId("reply-1")
                .waitExpiresAt(LocalDateTime.ofInstant(expiresAt, ZoneOffset.UTC))
                .deadlineAt(LocalDateTime.ofInstant(expiresAt, ZoneOffset.UTC))
                .agentStateSessionId("session-1")
                .nextSequence(9L)
                .build();
    }
}
>>>>>>> Stashed changes
