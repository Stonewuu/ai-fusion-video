package com.stonewu.fusion.controller.ai;

import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.GlobalExceptionHandler;
import com.stonewu.fusion.controller.ai.vo.AiChatReqVO;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.controller.ai.vo.PipelineRunStatusRespVO;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.security.SecurityUserDetails;
import com.stonewu.fusion.service.ai.agentscope.AgentScopeAssistantService;
import com.stonewu.fusion.service.ai.run.AgentRunQueryService;
import com.stonewu.fusion.service.ai.run.AgentRunReplayService;
import com.stonewu.fusion.service.ai.run.CancellationCoordinator;
import com.stonewu.fusion.service.ai.run.PipelineCursorParser;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.time.Instant;
import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.asyncDispatch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.request;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class AiPipelineSseControllerTests {

    private static final long CURRENT_USER_ID = 42L;

    private AgentScopeAssistantService assistant;
    private AgentRunQueryService queries;
    private AgentRunReplayService replay;
    private CancellationCoordinator cancellations;
    private AiPipelineController controller;
    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        assistant = mock(AgentScopeAssistantService.class);
        queries = mock(AgentRunQueryService.class);
        replay = mock(AgentRunReplayService.class);
        cancellations = mock(CancellationCoordinator.class);
        controller = new AiPipelineController(
                assistant,
                queries,
                replay,
                new PipelineCursorParser(),
                cancellations);
        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
        SecurityUserDetails user = new SecurityUserDetails(
                CURRENT_USER_ID, "owner", "secret", 1, null, List.of());
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken(
                        user, null, user.getAuthorities()));
    }

    @AfterEach
    void clearSecurityContext() {
        SecurityContextHolder.clearContext();
    }

    @Test
    void reconnectWritesStandardIdEventAndDataFrames() throws Exception {
        AgentRun run = run("run-1", "conversation-1", AgentRunStatus.RUNNING);
        CommittedAgentEvent committed = event("run-1", 8, "CONTENT", "hello");
        AiChatStreamRespVO projection = projection(
                "run-1", 8, "CONTENT", "hello", false);
        when(queries.resolveAuthorizedTarget(
                "run-1", null, CURRENT_USER_ID)).thenReturn(Mono.just(run));
        when(replay.replayThenLive("run-1", 7)).thenReturn(Flux.just(committed));
        when(queries.project(run, committed)).thenReturn(Mono.just(projection));

        MvcResult result = dispatch(mockMvc.perform(get(
                        "/api/ai/pipeline/reconnect")
                .param("runId", "run-1")
                .param("afterSequence", "7")
                .header("Last-Event-ID", "run-1:7")));

        String wire = result.getResponse().getContentAsString();
        assertThat(wire)
                .contains("id:run-1:8")
                .contains("event:pipeline-event")
                .contains("data:")
                .contains("\"runId\":\"run-1\"")
                .contains("\"sequence\":8")
                .contains("\"content\":\"hello\"");
        verify(replay).replayThenLive("run-1", 7);
    }

    @Test
    void reconnectRejectsConflictingHeaderAndQueryCursor() throws Exception {
        AgentRun run = run("run-1", "conversation-1", AgentRunStatus.RUNNING);
        when(queries.resolveAuthorizedTarget(
                "run-1", null, CURRENT_USER_ID)).thenReturn(Mono.just(run));

        MvcResult pending = mockMvc.perform(get("/api/ai/pipeline/reconnect")
                        .param("runId", "run-1")
                        .param("afterSequence", "7")
                        .header("Last-Event-ID", "run-1:8"))
                .andExpect(request().asyncStarted())
                .andReturn();
        mockMvc.perform(asyncDispatch(pending))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value(400));
        verify(replay, never()).replayThenLive("run-1", 7);
    }

    @Test
    void reconnectRejectsHeaderForAnotherRun() throws Exception {
        AgentRun run = run("run-1", "conversation-1", AgentRunStatus.RUNNING);
        when(queries.resolveAuthorizedTarget(
                "run-1", null, CURRENT_USER_ID)).thenReturn(Mono.just(run));

        MvcResult pending = mockMvc.perform(get("/api/ai/pipeline/reconnect")
                        .param("runId", "run-1")
                        .header("Last-Event-ID", "run-2:0"))
                .andExpect(request().asyncStarted())
                .andReturn();
        mockMvc.perform(asyncDispatch(pending))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value(400));
        verify(replay, never()).replayThenLive("run-1", 0);
    }

    @Test
    void crossUserRunIsHiddenAsHttp404() throws Exception {
        when(queries.resolveAuthorizedTarget(
                "foreign-run", null, CURRENT_USER_ID))
                .thenReturn(Mono.error(new BusinessException(
                        404, "Agent run does not exist")));

        MvcResult pending = mockMvc.perform(get("/api/ai/pipeline/reconnect")
                        .param("runId", "foreign-run"))
                .andExpect(request().asyncStarted())
                .andReturn();
        mockMvc.perform(asyncDispatch(pending))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.code").value(404));
    }

    @Test
    void terminalRunStillReplaysItsTerminalFrame() throws Exception {
        AgentRun run = run("run-terminal", "conversation-terminal",
                AgentRunStatus.COMPLETED);
        CommittedAgentEvent committed = event(
                "run-terminal", 3, "DONE", null);
        AiChatStreamRespVO projection = projection(
                "run-terminal", 3, "DONE", null, true);
        when(queries.resolveAuthorizedTarget(
                "run-terminal", null, CURRENT_USER_ID)).thenReturn(Mono.just(run));
        when(replay.replayThenLive("run-terminal", 0))
                .thenReturn(Flux.just(committed));
        when(queries.project(run, committed)).thenReturn(Mono.just(projection));

        MvcResult result = dispatch(mockMvc.perform(get(
                        "/api/ai/pipeline/reconnect")
                .param("runId", "run-terminal")));

        assertThat(result.getResponse().getContentAsString())
                .contains("id:run-terminal:3")
                .contains("\"outputType\":\"DONE\"")
                .contains("\"finished\":true");
    }

    @Test
    void startAuthorizesConversationBeforeOpeningObserverAndAddsIds() throws Exception {
        AiChatStreamRespVO projection = projection(
                "run-start", 1, "CONTENT", "started", false);
        when(queries.authorizeConversationForStart(
                "conversation-start", CURRENT_USER_ID)).thenReturn(Mono.empty());
        when(assistant.stream(
                org.mockito.ArgumentMatchers.any(AiChatReqVO.class),
                org.mockito.ArgumentMatchers.eq(CURRENT_USER_ID)))
                .thenReturn(Flux.just(projection));

        MvcResult result = dispatch(mockMvc.perform(post("/api/ai/pipeline/run")
                .contentType("application/json")
                .content("{\"conversationId\":\"conversation-start\",\"message\":\"hi\"}")));

        assertThat(result.getResponse().getContentAsString())
                .contains("id:run-start:1")
                .contains("data:");
        verify(queries).authorizeConversationForStart(
                "conversation-start", CURRENT_USER_ID);
    }

    @Test
    void cancelAndStatusUseTheCurrentUserAuthorizedRun() {
        AgentRun run = run("run-ops", "conversation-ops", AgentRunStatus.RUNNING);
        when(queries.resolveAuthorizedTarget(
                "run-ops", null, CURRENT_USER_ID)).thenReturn(Mono.just(run));
        when(cancellations.cancel("run-ops", CURRENT_USER_ID))
                .thenReturn(Mono.just(AgentRunStatus.CANCEL_REQUESTED));
        PipelineRunStatusRespVO status = new PipelineRunStatusRespVO(
                "run-ops", "CANCEL_REQUESTED", 4, null, null);
        when(queries.status("run-ops", null, CURRENT_USER_ID))
                .thenReturn(Mono.just(status));

        StepVerifier.create(controller.cancel("run-ops", null))
                .assertNext(response -> assertThat(response.getData()).isTrue())
                .verifyComplete();
        StepVerifier.create(controller.getStatus("run-ops", null))
                .assertNext(response -> assertThat(response.getData())
                        .isEqualTo(status))
                .verifyComplete();

        verify(cancellations).cancel("run-ops", CURRENT_USER_ID);
        verify(queries).status("run-ops", null, CURRENT_USER_ID);
    }

    private MvcResult dispatch(
            org.springframework.test.web.servlet.ResultActions request)
            throws Exception {
        MvcResult pending = request
                .andExpect(request().asyncStarted())
                .andReturn();
        return mockMvc.perform(asyncDispatch(pending))
                .andExpect(status().isOk())
                .andExpect(content().contentTypeCompatibleWith(
                        "text/event-stream"))
                .andReturn();
    }

    private AgentRun run(
            String runId, String conversationId, AgentRunStatus status) {
        return AgentRun.builder()
                .runId(runId)
                .conversationId(conversationId)
                .userId(CURRENT_USER_ID)
                .status(status.name())
                .nextSequence(9L)
                .deadlineAt(LocalDateTime.now().plusMinutes(5))
                .startedAt(LocalDateTime.now())
                .build();
    }

    private CommittedAgentEvent event(
            String runId,
            long sequence,
            String outputType,
            String content) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", outputType)
                .put("finished", "DONE".equals(outputType));
        if (content != null) {
            payload.put("content", content);
        }
        AgentEventEnvelope envelope = new AgentEventEnvelope(
                "raw-" + sequence,
                "DONE".equals(outputType) ? "AGENT_END" : "TEXT_BLOCK_DELTA",
                "main",
                null,
                null,
                null,
                null,
                null,
                outputType,
                payload,
                Instant.parse("2026-07-21T12:00:00Z"));
        return new CommittedAgentEvent(
                sequence, runId, sequence, envelope,
                Instant.parse("2026-07-21T12:00:01Z"));
    }

    private AiChatStreamRespVO projection(
            String runId,
            long sequence,
            String outputType,
            String content,
            boolean finished) {
        return new AiChatStreamRespVO()
                .setSchemaVersion(1)
                .setRunId(runId)
                .setSequence(sequence)
                .setOutputType(outputType)
                .setContent(content)
                .setFinished(finished);
    }
}
