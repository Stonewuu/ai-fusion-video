package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentRuntimeErrorCode;
import com.stonewu.fusion.enums.ai.AgentTerminalOutputType;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.AgentMessageService;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import com.stonewu.fusion.service.ai.run.AgentEventJournal;
import com.stonewu.fusion.service.ai.run.AgentMessageProjectionService;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.RunTerminalCoordinator;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.time.Instant;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.groups.Tuple.tuple;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@Testcontainers(disabledWithoutDocker = false)
class AgentProjectionRecoveryIT {

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
        properties.add("fusion.agentscope.v2.state.mode", () -> "in-memory");
    }

    @Autowired
    private AgentMessageProjectionService projectionService;

    @Autowired
    private AgentEventJournal journal;

    @Autowired
    private RunTerminalCoordinator terminalCoordinator;

    @Autowired
    private AgentRunCoordinator runCoordinator;

    @Autowired
    private AgentConversationService conversationService;

    @Autowired
    private AgentMessageService messageService;

    @Autowired
    private AgentMessageMapper messageMapper;

    @Autowired
    private AgentRunMapper runMapper;

    @Test
    void projectsAssistantAndToolHistoryInOrderAndCompletesTerminalCursor() {
        StartedAgentRun run = startRoot("projection-history");
        append(run, "THINKING_BLOCK_DELTA", "REASONING", null,
                objectPayload("delta", "先想"));
        append(run, "TEXT_BLOCK_DELTA", "CONTENT", null,
                objectPayload("delta", "第一段")
                        .put("reasoningDurationMs", 2500L));
        append(run, "TOOL_CALL_DELTA", null, "tool-1",
                objectPayload("delta", "{\"city\":"));
        append(run, "TOOL_CALL_DELTA", null, "tool-1",
                objectPayload("delta", "\"上海\"}"));
        append(run, "TOOL_CALL_END", "TOOL_CALL", "tool-1",
                objectPayload("toolCallName", "weather"));
        append(run, "TOOL_RESULT_TEXT_DELTA", null, "tool-1",
                objectPayload("delta", "晴天"));
        append(run, "TOOL_RESULT_END", "TOOL_FINISHED", "tool-1",
                objectPayload("toolCallName", "weather").put("state", "SUCCESS"));
        append(run, "TEXT_BLOCK_DELTA", "CONTENT", null,
                objectPayload("delta", "最终答案"));
        assertThat(await(terminalCoordinator.terminateOwned(
                completedTerminal(run),
                run.ownerInstanceId(),
                run.ownerEpoch())))
                .isPresent();
        long terminalSequence = requireRun(run.runId()).getTerminalSequence();

        await(projectionService.projectThrough(run.runId(), terminalSequence));
        await(projectionService.projectThrough(run.runId(), terminalSequence));

        List<AgentMessage> projected = projectedMessages(run.runId());
        assertThat(projected).hasSize(4);
        assertThat(projected.get(0).getRole()).isEqualTo("assistant");
        assertThat(projected.get(0).getReasoningContent()).isEqualTo("先想");
        assertThat(projected.get(0).getReasoningDurationMs()).isEqualTo(2500L);
        assertThat(projected.get(0).getContent()).isEqualTo("第一段");
        assertThat(projected.get(1).getRole()).isEqualTo("tool");
        assertThat(projected.get(1).getToolStatus()).isEqualTo("running");
        assertThat(projected.get(1).getToolName()).isEqualTo("weather");
        assertThat(projected.get(1).getToolCallId()).isEqualTo("tool-1");
        assertThat(projected.get(1).getContent()).isEqualTo("{\"city\":\"上海\"}");
        assertThat(projected.get(2).getRole()).isEqualTo("tool");
        assertThat(projected.get(2).getToolStatus()).isEqualTo("success");
        assertThat(projected.get(2).getContent()).isEqualTo("晴天");
        assertThat(projected.get(3).getRole()).isEqualTo("assistant");
        assertThat(projected.get(3).getContent()).isEqualTo("最终答案");
        assertThat(projected)
                .extracting(AgentMessage::getProjectionKey)
                .allSatisfy(key -> assertThat(key).matches("[0-9a-f]{64}"))
                .doesNotHaveDuplicates();
        assertThat(projected)
                .extracting(AgentMessage::getMessageOrder)
                .containsExactly(2L, 3L, 4L, 5L);

        AgentRun persisted = requireRun(run.runId());
        assertThat(persisted.getProjectedThroughSequence()).isEqualTo(terminalSequence);
        assertThat(persisted.getProjectionCompletedAt()).isNotNull();
        assertThat(conversationService.getByConversationId(run.conversationId()).getStatus())
                .isEqualTo("completed");
    }

    @Test
    void resumesAfterCommittedInsertBeforeCursorWithoutDuplicatingMessage() {
        StartedAgentRun run = startRoot("projection-crash-retry");
        append(run, "TEXT_BLOCK_DELTA", "CONTENT", null,
                objectPayload("delta", "可恢复内容"));
        assertThat(await(terminalCoordinator.terminateOwned(
                completedTerminal(run),
                run.ownerInstanceId(),
                run.ownerEpoch())))
                .isPresent();
        long terminalSequence = requireRun(run.runId()).getTerminalSequence();
        String key = projectionKey(run.runId(), terminalSequence, "assistant");
        messageService.saveProjectedMessage(
                requireRun(run.runId()).getConversationId(),
                AgentMessage.builder()
                        .runId(run.runId())
                        .projectionKey(key)
                        .role("assistant")
                        .content("可恢复内容")
                        .build());
        assertThat(requireRun(run.runId()).getProjectedThroughSequence()).isZero();

        await(Mono.when(
                projectionService.projectThrough(run.runId(), terminalSequence),
                projectionService.projectThrough(run.runId(), terminalSequence)));

        assertThat(projectedMessages(run.runId()))
                .singleElement()
                .satisfies(message -> {
                    assertThat(message.getProjectionKey()).isEqualTo(key);
                    assertThat(message.getContent()).isEqualTo("可恢复内容");
                });
        AgentRun persisted = requireRun(run.runId());
        assertThat(persisted.getProjectedThroughSequence()).isEqualTo(terminalSequence);
        assertThat(persisted.getProjectionCompletedAt()).isNotNull();
    }

    @Test
    void persistsOfficialAgentScopeErrorBlockAsErrorDespiteSuccessEndState() {
        StartedAgentRun run = startRoot("projection-tool-error");
        append(run, "TOOL_RESULT_TEXT_DELTA", null, "tool-error",
                objectPayload("delta", "Error: Tool execution failed: invalid input"));
        append(run, "TOOL_RESULT_END", "TOOL_FINISHED", "tool-error",
                objectPayload("toolName", "update_script_info").put("state", "SUCCESS"));
        assertThat(await(terminalCoordinator.terminateOwned(
                completedTerminal(run),
                run.ownerInstanceId(),
                run.ownerEpoch())))
                .isPresent();
        long terminalSequence = requireRun(run.runId()).getTerminalSequence();

        await(projectionService.projectThrough(run.runId(), terminalSequence));

        assertThat(projectedMessages(run.runId()))
                .singleElement()
                .satisfies(message -> {
                    assertThat(message.getRole()).isEqualTo("tool");
                    assertThat(message.getToolStatus()).isEqualTo("error");
                    assertThat(message.getContent())
                            .isEqualTo("Error: Tool execution failed: invalid input");
                });
    }

    @Test
    void cancellationMarksOnlyToolsWithoutAResultAsCancelled() {
        StartedAgentRun run = startRoot("projection-cancel-tools");
        append(run, "TOOL_CALL_END", "TOOL_CALL", "finished-tool",
                objectPayload("toolCallName", "finished_tool"));
        append(run, "TOOL_RESULT_END", "TOOL_FINISHED", "finished-tool",
                objectPayload("toolCallName", "finished_tool").put("state", "SUCCESS"));
        append(run, "TOOL_CALL_END", "TOOL_CALL", "unfinished-tool",
                objectPayload("toolCallName", "unfinished_tool"));
        assertThat(await(terminalCoordinator.terminateOwned(
                cancelledTerminal(run),
                run.ownerInstanceId(),
                run.ownerEpoch())))
                .isPresent();
        long terminalSequence = requireRun(run.runId()).getTerminalSequence();

        await(projectionService.projectThrough(run.runId(), terminalSequence));
        await(projectionService.projectThrough(run.runId(), terminalSequence));

        assertThat(projectedMessages(run.runId()))
                .extracting(
                        AgentMessage::getToolCallId,
                        AgentMessage::getToolStatus)
                .containsExactly(
                        tuple("finished-tool", "running"),
                        tuple("finished-tool", "success"),
                        tuple("unfinished-tool", "cancelled"));
        assertThat(conversationService.getByConversationId(run.conversationId()).getStatus())
                .isEqualTo("cancelled");
    }

    @Test
    void rejectedConfirmationProjectsATerminalToolHistoryState() {
        StartedAgentRun run = startRoot("projection-rejected-tool");
        append(run, "TOOL_CALL_DELTA", null, "rejected-tool",
                objectPayload("delta", "{\"scriptId\":1}"));
        append(run, "TOOL_CALL_END", "TOOL_CALL", "rejected-tool",
                objectPayload("toolCallName", "update_script_info"));
        ObjectNode confirmation = JsonNodeFactory.instance.objectNode();
        confirmation.putArray("pendingToolCalls")
                .addObject()
                .put("toolCallId", "rejected-tool")
                .put("toolName", "update_script_info")
                .put("argumentsPreview", "{\"scriptId\":1}");
        confirmation.putArray("decisions")
                .addObject()
                .put("toolCallId", "rejected-tool")
                .put("approved", false);
        append(run, "USER_CONFIRM_RESULT", "USER_CONFIRM_RESULT", null, confirmation);
        assertThat(await(terminalCoordinator.terminateOwned(
                completedTerminal(run),
                run.ownerInstanceId(),
                run.ownerEpoch())))
                .isPresent();
        long terminalSequence = requireRun(run.runId()).getTerminalSequence();

        await(projectionService.projectThrough(run.runId(), terminalSequence));

        assertThat(projectedMessages(run.runId()))
                .extracting(
                        AgentMessage::getToolCallId,
                        AgentMessage::getToolStatus)
                .containsExactly(
                        tuple("rejected-tool", "running"),
                        tuple("rejected-tool", "rejected"));
    }

    @Test
    void mirroredChildEventsAdvanceTheParentCursorWithoutDuplicatingHistory() {
        StartedAgentRun run = startRoot("projection-mirrored-child");
        append(run, "TOOL_CALL_END", "TOOL_CALL", "child-inner-tool",
                objectPayload("toolCallName", "child_tool")
                        .put("_platformMirroredChildEvent", true)
                        .put("childRunId", "child-run-1")
                        .put("childSequence", 4));
        assertThat(await(terminalCoordinator.terminateOwned(
                completedTerminal(run),
                run.ownerInstanceId(),
                run.ownerEpoch())))
                .isPresent();
        long terminalSequence = requireRun(run.runId()).getTerminalSequence();

        await(projectionService.projectThrough(run.runId(), terminalSequence));

        assertThat(projectedMessages(run.runId())).isEmpty();
        AgentRun persisted = requireRun(run.runId());
        assertThat(persisted.getProjectedThroughSequence()).isEqualTo(terminalSequence);
        assertThat(persisted.getProjectionCompletedAt()).isNotNull();
    }

    @Test
    void conflictingProjectionFailsClosedAndDoesNotAdvanceCursor() {
        StartedAgentRun run = startRoot("projection-conflict");
        append(run, "TEXT_BLOCK_DELTA", "CONTENT", null,
                objectPayload("delta", "真实内容"));
        assertThat(await(terminalCoordinator.terminateOwned(
                completedTerminal(run),
                run.ownerInstanceId(),
                run.ownerEpoch())))
                .isPresent();
        long terminalSequence = requireRun(run.runId()).getTerminalSequence();
        messageService.saveProjectedMessage(
                requireRun(run.runId()).getConversationId(),
                AgentMessage.builder()
                        .runId(run.runId())
                        .projectionKey(projectionKey(
                                run.runId(), terminalSequence, "assistant"))
                        .role("assistant")
                        .content("冲突内容")
                        .build());

        assertThatThrownBy(() -> await(
                projectionService.projectThrough(run.runId(), terminalSequence)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage(
                        "Projection key resolves to a different Agent message: "
                                + projectionKey(
                                        run.runId(), terminalSequence, "assistant"));
        AgentRun persisted = requireRun(run.runId());
        assertThat(persisted.getProjectedThroughSequence()).isZero();
        assertThat(persisted.getProjectionCompletedAt()).isNull();
        assertThat(projectedMessages(run.runId())).hasSize(1);
    }

    private void append(
            StartedAgentRun run,
            String rawEventType,
            String outputType,
            String toolCallId,
            ObjectNode payload) {
        if (outputType != null) {
            payload.put("outputType", outputType);
        }
        assertThat(await(journal.appendOwned(
                run.runId(),
                run.ownerInstanceId(),
                run.ownerEpoch(),
                new AgentEventEnvelope(
                        uniqueId("projection-event"),
                        rawEventType,
                        "assistant",
                        "reply-1",
                        null,
                        toolCallId,
                        null,
                        null,
                        outputType,
                        payload,
                        Instant.now()))))
                .isPresent();
    }

    private RunTerminalRequest completedTerminal(StartedAgentRun run) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", "DONE")
                .put("finished", true);
        return new RunTerminalRequest(
                run.runId(),
                new StateStoreSlot("42", run.agentStateSessionId()),
                Set.of(AgentRunStatus.RUNNING),
                AgentRunStatus.COMPLETED,
                AgentTerminalOutputType.DONE,
                null,
                null,
                new AgentEventEnvelope(
                        uniqueId("projection-terminal"),
                        "AGENT_END",
                        "assistant",
                        null,
                        null,
                        null,
                        null,
                        null,
                        "DONE",
                        payload,
                        Instant.now()));
    }

    private RunTerminalRequest cancelledTerminal(StartedAgentRun run) {
        ObjectNode payload = JsonNodeFactory.instance.objectNode()
                .put("outputType", "CANCELLED")
                .put("finished", true)
                .put("errorCode", AgentRuntimeErrorCode.RUN_CANCELLED.name());
        return new RunTerminalRequest(
                run.runId(),
                new StateStoreSlot("42", run.agentStateSessionId()),
                Set.of(AgentRunStatus.RUNNING),
                AgentRunStatus.CANCELLED,
                AgentTerminalOutputType.CANCELLED,
                AgentRuntimeErrorCode.RUN_CANCELLED,
                "用户已取消",
                new AgentEventEnvelope(
                        uniqueId("projection-terminal"),
                        "REQUEST_STOP",
                        "assistant",
                        null,
                        null,
                        null,
                        null,
                        null,
                        "CANCELLED",
                        payload,
                        Instant.now()));
    }

    private ObjectNode objectPayload(String field, String value) {
        return JsonNodeFactory.instance.objectNode().put(field, value);
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
                "Projection test",
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
                        "Projection test agent",
                        "You are a projection test agent.",
                        10,
                        "model-config-1",
                        1L,
                        "openai",
                        "gpt-test",
                        JsonNodeFactory.instance.objectNode().put("temperature", 0.2),
                        List.of(),
                        "1.0.0-test"));
    }

    private List<AgentMessage> projectedMessages(String runId) {
        return messageMapper.selectList(new LambdaQueryWrapper<AgentMessage>()
                .eq(AgentMessage::getRunId, runId)
                .isNotNull(AgentMessage::getProjectionKey)
                .orderByAsc(AgentMessage::getMessageOrder));
    }

    private AgentRun requireRun(String runId) {
        AgentRun run = runMapper.selectOne(new LambdaQueryWrapper<AgentRun>()
                .eq(AgentRun::getRunId, runId));
        assertThat(run).isNotNull();
        return run;
    }

    private <T> T await(Mono<T> value) {
        // Blocking is restricted to this integration-test boundary.
        return value.block(Duration.ofSeconds(30));
    }

    private static String projectionKey(String runId, long sequence, String kind) {
        String material = runId + ':' + sequence + ':' + kind;
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(material.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException(impossible);
        }
    }

    private static String uniqueId(String prefix) {
        String boundedPrefix = prefix.length() <= 24 ? prefix : prefix.substring(0, 24);
        return boundedPrefix + '-' + UUID.randomUUID().toString().replace("-", "");
    }
}
