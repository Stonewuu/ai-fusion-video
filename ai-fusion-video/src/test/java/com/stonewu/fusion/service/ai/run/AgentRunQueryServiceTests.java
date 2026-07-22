package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import reactor.core.scheduler.Schedulers;
import reactor.test.StepVerifier;

import java.time.Instant;
import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AgentRunQueryServiceTests {

    private AgentRunMapper runMapper;
    private AgentEventMapper eventMapper;
    private AgentRunQueryService service;

    @BeforeEach
    void setUp() {
        runMapper = mock(AgentRunMapper.class);
        eventMapper = mock(AgentEventMapper.class);
        AgentRuntimeSchedulers schedulers = mock(AgentRuntimeSchedulers.class);
        when(schedulers.journal()).thenReturn(Schedulers.immediate());
        ObjectMapper objectMapper = new ObjectMapper()
                .findAndRegisterModules()
                .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
        service = new AgentRunQueryService(
                runMapper,
                mock(AgentConversationMapper.class),
                eventMapper,
                mock(AgentEventRepository.class),
                objectMapper,
                schedulers);
    }

    @Test
    void restoresLegacyContentAndOverwritesIdentityFromCommittedEnvelope() {
        AgentRun run = run();
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("runId", "forged-run")
                .put("sequence", 999)
                .put("delta", "hello");
        CommittedAgentEvent event = committed(
                7, "CONTENT", "TEXT_BLOCK_DELTA", null, payload);

        StepVerifier.create(service.project(run, event))
                .assertNext(projected -> {
                    assertThat(projected.getSchemaVersion()).isEqualTo(1);
                    assertThat(projected.getRunId()).isEqualTo("run-1");
                    assertThat(projected.getSequence()).isEqualTo(7L);
                    assertThat(projected.getConversationId())
                            .isEqualTo("conversation-1");
                    assertThat(projected.getOutputType()).isEqualTo("CONTENT");
                    assertThat(projected.getContent()).isEqualTo("hello");
                    assertThat(projected.getRawEventType())
                            .isEqualTo("TEXT_BLOCK_DELTA");
                    assertThat(projected.getFinished()).isFalse();
                })
                .verifyComplete();
    }

    @Test
    void reconstructsToolArgumentsFromRawJournalDeltas() {
        AgentRun run = run();
        AgentEvent delta = AgentEvent.builder()
                .id(41L)
                .runId("run-1")
                .sequenceNo(5L)
                .rawEventType("TOOL_CALL_DELTA")
                .payloadJson("{\"delta\":\"{\\\"assetId\\\":7}\"}")
                .build();
        when(eventMapper.selectToolDeltas("run-1", "tool-1", 6))
                .thenReturn(List.of(delta));
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("toolCallName", "asset_lookup");
        CommittedAgentEvent event = committed(
                6, "TOOL_CALL", "TOOL_CALL_END", "tool-1", payload);

        StepVerifier.create(service.project(run, event))
                .assertNext(projected -> {
                    assertThat(projected.getToolCallId()).isEqualTo("tool-1");
                    assertThat(projected.getToolName()).isEqualTo("asset_lookup");
                    assertThat(projected.getToolCalls())
                            .singleElement()
                            .satisfies(tool -> {
                                assertThat(tool.getId()).isEqualTo("tool-1");
                                assertThat(tool.getName()).isEqualTo("asset_lookup");
                                assertThat(tool.getArguments())
                                        .isEqualTo("{\"assetId\":7}");
                            });
                })
                .verifyComplete();
    }

    @Test
    void projectsOfficialAgentScopeErrorBlockAsErrorDespiteSuccessEndState() {
        AgentRun run = run();
        AgentEvent delta = AgentEvent.builder()
                .id(42L)
                .runId("run-1")
                .sequenceNo(6L)
                .rawEventType("TOOL_RESULT_TEXT_DELTA")
                .payloadJson("{\"delta\":\"Error: Tool execution failed: invalid input\"}")
                .build();
        when(eventMapper.selectToolDeltas("run-1", "tool-1", 7))
                .thenReturn(List.of(delta));
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("toolCallName", "update_script_info")
                .put("state", "SUCCESS");
        CommittedAgentEvent event = committed(
                7, "TOOL_FINISHED", "TOOL_RESULT_END", "tool-1", payload);

        StepVerifier.create(service.project(run, event))
                .assertNext(projected -> {
                    assertThat(projected.getToolResult())
                            .isEqualTo("Error: Tool execution failed: invalid input");
                    assertThat(projected.getToolStatus()).isEqualTo("error");
                })
                .verifyComplete();
    }

    @Test
    void hidesMissingOrCrossUserRunAsNotFound() {
        when(runMapper.selectAuthorizedByRunId("foreign-run", 42L))
                .thenReturn(null);

        StepVerifier.create(service.requireAuthorizedRun("foreign-run", 42L))
                .expectErrorSatisfies(failure -> {
                    assertThat(failure).isInstanceOf(BusinessException.class);
                    assertThat(((BusinessException) failure).getCode()).isEqualTo(404);
                })
                .verify();
    }

    private AgentRun run() {
        return AgentRun.builder()
                .runId("run-1")
                .conversationId("conversation-1")
                .userId(42L)
                .status("RUNNING")
                .nextSequence(8L)
                .deadlineAt(LocalDateTime.now().plusMinutes(5))
                .startedAt(LocalDateTime.now())
                .build();
    }

    private CommittedAgentEvent committed(
            long sequence,
            String outputType,
            String rawEventType,
            String toolCallId,
            ObjectNode payload) {
        AgentEventEnvelope envelope = new AgentEventEnvelope(
                "raw-" + sequence,
                rawEventType,
                "main",
                "reply-1",
                "block-1",
                toolCallId,
                null,
                null,
                outputType,
                payload,
                Instant.parse("2026-07-21T12:00:00Z"));
        return new CommittedAgentEvent(
                sequence, "run-1", sequence, envelope,
                Instant.parse("2026-07-21T12:00:01Z"));
    }
}
