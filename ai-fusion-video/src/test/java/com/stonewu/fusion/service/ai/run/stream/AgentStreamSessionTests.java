package com.stonewu.fusion.service.ai.run.stream;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class AgentStreamSessionTests {
    private final ObjectMapper json = new ObjectMapper().findAndRegisterModules();
    private final AgentRedisStreamStore streams = mock(AgentRedisStreamStore.class);
    private final AgentStreamMessageRepository messages = mock(AgentStreamMessageRepository.class);
    private final AgentEventRepository controls = mock(AgentEventRepository.class);
    private final List<AgentMessage> saved = new ArrayList<>();
    private final List<AiChatStreamRespVO> published = new ArrayList<>();
    private int nextId;

    @Test
    void hundredsOfChunksStreamImmediatelyButPersistAsOneCompleteReply() {
        AgentStreamSession session = session(false);
        for (int index = 0; index < 300; index++) {
            session.accept(event("TEXT_DELTA", "CONTENT", "reply", null, Map.of("delta", "片段")));
        }
        assertThat(published).hasSize(300);
        assertThat(saved).isEmpty();
        session.accept(event("MODEL_CALL_END", null, "reply", null, Map.of()));
        session.finish();

        assertThat(saved).singleElement().satisfies(message -> {
            assertThat(message.getRole()).isEqualTo("assistant");
            assertThat(message.getContent()).isEqualTo("片段".repeat(300));
        });
        assertThat(published).hasSize(301);
        assertThat(published.getLast().getContentMode()).isEqualTo("replace");
        assertThat(published.getLast().getMessageId()).isEqualTo(saved.getFirst().getProjectionKey());
        verifyNoInteractions(controls);
    }

    @Test
    void toolExecutionAndFollowingReasoningWaitForCompleteArgumentsAndResult() {
        AgentStreamSession session = session(false);
        AtomicBoolean callReady = new AtomicBoolean();
        AtomicBoolean resultReady = new AtomicBoolean();
        session.awaitCall("tool").doOnSuccess(ignored -> callReady.set(true)).subscribe();
        session.awaitResult("tool").doOnSuccess(ignored -> resultReady.set(true)).subscribe();
        session.accept(event("TEXT_DELTA", "CONTENT", "reply", null, Map.of("delta", "正在查询")));
        session.accept(event("TOOL_CALL_START", "TOOL_CALL_STARTED", "reply", "tool", Map.of("toolCallName", "lookup")));
        session.accept(event("TOOL_CALL_DELTA", null, "reply", "tool", Map.of("delta", "{\"id\":")));
        session.accept(event("TOOL_CALL_DELTA", null, "reply", "tool", Map.of("delta", "7}")));
        session.accept(event("TOOL_CALL_END", "TOOL_CALL", "reply", "tool", Map.of()));
        assertThat(callReady).isFalse();
        assertThat(saved).isEmpty();
        session.accept(event("MODEL_CALL_END", null, "reply", null, Map.of()));
        assertThat(callReady).isTrue();
        assertThat(saved).hasSize(2);
        assertThat(saved.getLast().getContent()).isEqualTo("{\"id\":7}");
        assertThat(saved.getLast().getToolStatus()).isEqualTo("running");

        session.accept(event("TOOL_RESULT_TEXT_DELTA", null, "reply", "tool", Map.of("delta", "查到")));
        session.accept(event("TOOL_RESULT_TEXT_DELTA", null, "reply", "tool", Map.of("delta", "七条")));
        assertThat(resultReady).isFalse();
        session.accept(event("TOOL_RESULT_END", "TOOL_FINISHED", "reply", "tool", Map.of("state", "COMPLETED")));
        assertThat(resultReady).isTrue();
        assertThat(saved).hasSize(3);
        assertThat(saved.getLast().getContent()).isEqualTo("查到七条");
        assertThat(saved.getLast().getToolStatus()).isEqualTo("success");
        assertThat(published.getLast().getToolResult()).isEqualTo("查到七条");
        verifyNoInteractions(controls);
    }

    @Test
    void failedModelAttemptIsDiscardedAndCancellationSavesOnlyAcceptedPartialReply() {
        AgentStreamSession session = session(false);
        session.accept(event("TEXT_DELTA", "CONTENT", "failed", null, Map.of("delta", "失败草稿")));
        session.accept(event("MODEL_RETRY", "MODEL_RETRY", "failed", null, Map.of()));
        session.accept(event("TEXT_DELTA", "CONTENT", "retry", null, Map.of("delta", "取消前已显示")));
        session.finish();
        int sizeAtFinish = published.size();
        session.accept(event("TEXT_DELTA", "CONTENT", "retry", null, Map.of("delta", "取消后片段")));
        session.finish();
        assertThat(saved).singleElement().extracting(AgentMessage::getContent).isEqualTo("取消前已显示");
        assertThat(published).hasSize(sizeAtFinish);
    }

    @Test
    void userDenialDoesNotProduceAnotherSuccessfulToolResult() {
        AgentStreamSession session = session(false);
        session.accept(event("USER_CONFIRM_RESULT", null, "reply", null, Map.of("confirmResults",
                List.of(Map.of("confirmed", false, "toolCall", Map.of("id", "denied"))))));
        session.accept(event("TOOL_RESULT_TEXT_DELTA", null, "reply", "denied",
                Map.of("toolCallName", "save_project", "delta", "Permission denied by user")));
        session.accept(event("TOOL_RESULT_END", "TOOL_FINISHED", "reply", "denied", Map.of("state", "denied")));
        assertThat(saved).isEmpty();
        assertThat(published).isEmpty();
    }

    @Test
    void childPublishesIntoParentStreamAndSavesOnlyItsOwnMessage() {
        AgentStreamSession session = session(true);
        session.accept(event("TEXT_DELTA", "CONTENT", "child-reply", null, Map.of("delta", "子任务回复")));
        session.accept(event("MODEL_CALL_END", null, "child-reply", null, Map.of()));
        assertThat(saved).singleElement().satisfies(message -> {
            assertThat(message.getParentToolCallId()).isEqualTo("delegate");
            assertThat(message.getContent()).isEqualTo("子任务回复");
        });
        assertThat(published).allSatisfy(event -> {
            assertThat(event.getRunId()).isEqualTo("root");
            assertThat(event.getSourceRunId()).isEqualTo("child");
            assertThat(event.getParentToolCallId()).isEqualTo("delegate");
        });
        verifyNoInteractions(controls);
    }

    private AgentStreamSession session(boolean child) {
        doAnswer(invocation -> {
            saved.addAll(invocation.getArgument(3));
            return null;
        }).when(messages).save(anyString(), anyString(), anyLong(), anyList());
        when(streams.append(anyString(), anyString(), any())).thenAnswer(invocation -> {
            AiChatStreamRespVO event = invocation.getArgument(2);
            event.setRunId(invocation.getArgument(0));
            published.add(event);
            return event;
        });
        AgentRun run = AgentRun.builder().runId(child ? "child" : "root").streamRunId("root")
                .streamProtocolVersion(2).conversationId("conversation")
                .deadlineAt(LocalDateTime.of(2099, 1, 1, 0, 0))
                .parentToolCallId(child ? "delegate" : null).agentName(child ? "writer" : null).build();
        return new AgentStreamSession(run, "owner", 1, streams, messages,
                new AgentStreamEventFormatter(json), controls, json);
    }

    private AgentEventEnvelope event(String type, String output, String reply, String tool, Map<String, ?> payload) {
        return new AgentEventEnvelope("event-" + ++nextId, type, "agent", reply, null, tool,
                null, null, output, json.valueToTree(payload), Instant.now());
    }
}
