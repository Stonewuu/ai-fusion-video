package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.BinaryNode;
import com.fasterxml.jackson.databind.node.TextNode;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import io.agentscope.core.event.AgentEndEvent;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.event.AgentResultEvent;
import io.agentscope.core.event.AgentStartEvent;
import io.agentscope.core.event.AllToolsDeniedEvent;
import io.agentscope.core.event.ConfirmResult;
import io.agentscope.core.event.CustomEvent;
import io.agentscope.core.event.DataBlockDeltaEvent;
import io.agentscope.core.event.DataBlockEndEvent;
import io.agentscope.core.event.DataBlockStartEvent;
import io.agentscope.core.event.ExceedMaxItersEvent;
import io.agentscope.core.event.ExternalExecutionResultEvent;
import io.agentscope.core.event.HintBlockEvent;
import io.agentscope.core.event.ModelCallEndEvent;
import io.agentscope.core.event.ModelCallStartEvent;
import io.agentscope.core.event.RequestStopEvent;
import io.agentscope.core.event.RequireExternalExecutionEvent;
import io.agentscope.core.event.RequireUserConfirmEvent;
import io.agentscope.core.event.SubagentExposedEvent;
import io.agentscope.core.event.TextBlockDeltaEvent;
import io.agentscope.core.event.TextBlockEndEvent;
import io.agentscope.core.event.TextBlockStartEvent;
import io.agentscope.core.event.ThinkingBlockDeltaEvent;
import io.agentscope.core.event.ThinkingBlockEndEvent;
import io.agentscope.core.event.ThinkingBlockStartEvent;
import io.agentscope.core.event.ToolCallDeltaEvent;
import io.agentscope.core.event.ToolCallEndEvent;
import io.agentscope.core.event.ToolCallStartEvent;
import io.agentscope.core.event.ToolResultDataDeltaEvent;
import io.agentscope.core.event.ToolResultEndEvent;
import io.agentscope.core.event.ToolResultStartEvent;
import io.agentscope.core.event.ToolResultTextDeltaEvent;
import io.agentscope.core.event.UserConfirmResultEvent;
import io.agentscope.core.message.GenerateReason;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolResultState;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.model.ChatUsage;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class AgentScopeEventMapperTests {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final AgentEventEnvelopeSanitizer sanitizer = new AgentEventEnvelopeSanitizer();
    private final AgentScopeEventMapper mapper = new AgentScopeEventMapper(objectMapper, sanitizer);

    @Test
    void derivesIdentityForEveryGaEventSubtypeAndCoversAllThirtyOneTypes() {
        List<IdentityCase> cases = identityCases();

        assertThat(io.agentscope.core.event.AgentEventType.values()).hasSize(31);
        assertThat(cases).hasSize(31);
        assertThat(cases)
                .extracting(testCase -> testCase.event().getType())
                .containsExactlyInAnyOrder(io.agentscope.core.event.AgentEventType.values());

        for (IdentityCase testCase : cases) {
            AgentEventEnvelope envelope = mapper.map(testCase.event());
            assertThat(envelope.replyId())
                    .as(testCase.event().getType() + " replyId")
                    .isEqualTo(testCase.replyId());
            assertThat(envelope.blockId())
                    .as(testCase.event().getType() + " blockId")
                    .isEqualTo(testCase.blockId());
            assertThat(envelope.toolCallId())
                    .as(testCase.event().getType() + " toolCallId")
                    .isEqualTo(testCase.toolCallId());
        }
    }

    @Test
    void preservesRawIdentitySourceAndCreatedAtFromTheRealEvent() {
        TextBlockDeltaEvent source = new TextBlockDeltaEvent(
                "raw-text-1",
                "2026-07-21T12:34:56.123Z",
                "reply-1",
                "block-1",
                "你好");
        source.withSource("main/researcher");
        source.withMetadata(Map.of("parentToolCallId", "parent-tool-1"));

        AgentEventEnvelope envelope = mapper.map(source);

        assertThat(envelope.rawEventId()).isEqualTo("raw-text-1");
        assertThat(envelope.rawEventType()).isEqualTo("TEXT_BLOCK_DELTA");
        assertThat(envelope.source()).isEqualTo("main/researcher");
        assertThat(envelope.replyId()).isEqualTo("reply-1");
        assertThat(envelope.blockId()).isEqualTo("block-1");
        assertThat(envelope.toolCallId()).isNull();
        assertThat(envelope.parentToolCallId()).isEqualTo("parent-tool-1");
        assertThat(envelope.agentName()).isEqualTo("researcher");
        assertThat(envelope.outputType()).isEqualTo("CONTENT");
        assertThat(envelope.createdAt()).isEqualTo(Instant.parse("2026-07-21T12:34:56.123Z"));
        assertThat(envelope.payload().path("delta").asText()).isEqualTo("你好");
    }

    @Test
    void keepsMainLifecycleEndRawOnlyAndClearsParentAndAgentIdentity() {
        AgentEndEvent source = new AgentEndEvent(
                "raw-end-1", "2026-07-21T12:35:00Z", "reply-end-1");
        source.withMetadata(Map.of(
                "parentToolCallId", "must-not-leak",
                "agentName", "must-not-leak"));

        AgentEventEnvelope envelope = mapper.map(source);

        assertThat(envelope.replyId()).isEqualTo("reply-end-1");
        assertThat(envelope.parentToolCallId()).isNull();
        assertThat(envelope.agentName()).isNull();
        assertThat(envelope.outputType()).isNull();

        Msg result = Msg.builder()
                .name("main-agent")
                .role(MsgRole.ASSISTANT)
                .textContent("done")
                .build();
        AgentEventEnvelope resultEnvelope = mapper.map(
                new AgentResultEvent(result).withMetadata(Map.of(
                        "parentToolCallId", "must-not-leak",
                        "agentName", "must-not-leak")));
        assertThat(resultEnvelope.parentToolCallId()).isNull();
        assertThat(resultEnvelope.agentName()).isNull();
    }

    @Test
    void mapsOnlyStableFrontendProjectionTypes() {
        assertThat(mapper.map(new TextBlockDeltaEvent("r", "b", "text")).outputType())
                .isEqualTo("CONTENT");
        assertThat(mapper.map(new ThinkingBlockDeltaEvent("r", "b", "thought")).outputType())
                .isEqualTo("REASONING");
        assertThat(mapper.map(new ToolCallEndEvent("r", "c", "tool")).outputType())
                .isEqualTo("TOOL_CALL");
        assertThat(mapper.map(new ToolResultEndEvent(
                "r", "c", "tool", ToolResultState.SUCCESS)).outputType())
                .isEqualTo("TOOL_FINISHED");
        assertThat(mapper.map(new SubagentExposedEvent(
                "sub", "researcher", "session", "Researcher")).outputType())
                .isEqualTo("SUB_AGENT_STARTED");
        assertThat(mapper.map(new AgentEndEvent("r").withSource("main/researcher")).outputType())
                .isEqualTo("SUB_AGENT_FINISHED");
        assertThat(mapper.map(new RequireUserConfirmEvent("r", List.of())).outputType())
                .isEqualTo("USER_CONFIRMATION_REQUIRED");
        assertThat(mapper.map(new RequireExternalExecutionEvent("r", List.of())).outputType())
                .isEqualTo("EXTERNAL_EXECUTION_REQUIRED");
        assertThat(mapper.map(new UserConfirmResultEvent("r", List.of())).outputType())
                .isEqualTo("USER_CONFIRM_RESULT");
        assertThat(mapper.map(new ExternalExecutionResultEvent("r", List.of())).outputType())
                .isEqualTo("EXTERNAL_EXECUTION_RESULT");
        assertThat(mapper.map(new AgentStartEvent("session", "r", "main")).outputType())
                .isNull();
    }

    @Test
    void sanitizesSensitivePayloadWithoutDroppingTheEvent() {
        CustomEvent source = new CustomEvent(
                "raw-custom-1",
                "2026-07-21T12:36:00Z",
                "unsafe_payload",
                Map.of(
                        "Authorization", "Bearer abcdefgh123456",
                        "credential", "credential-value",
                        "binary", "aGVsbG8=",
                        "dataUri", "data:text/plain,hello",
                        "signed", "https://media.test/a?X-Amz-Signature=abcdef",
                        "encodedSigned", "https://media.test/a?X-Amz-Signature%3Dabcdef",
                        "path", "C:\\Users\\admin\\secret.txt",
                        "safe", "preserved"));

        AgentEventEnvelope envelope = mapper.map(source);
        String payloadJson = envelope.payloadJson(objectMapper);

        assertThat(envelope.rawEventType()).isEqualTo("CUSTOM");
        assertThat(payloadJson)
                .contains("[SECRET_REDACTED]", "[BINARY_REDACTED]",
                        "[SIGNED_URL_REDACTED]", "[FILE_PATH_REDACTED]", "preserved")
                .doesNotContain("abcdefgh123456", "credential-value", "aGVsbG8=",
                        "X-Amz-Signature", "Users\\\\admin");
    }

    @Test
    void preservesPublicMediaUrlsAndTheirPossibleStreamFragments() {
        assertThat(List.of(
                "http://localhost:3000",
                "/media",
                "/images",
                "/6fd59b7ad74b42be9d87c823a78c615f.png",
                "http://localhost:3000/media/images/6fd59b7ad74b42be9d87c823a78c615f.png"))
                .allSatisfy(value -> assertThat(sanitizer.sanitize(TextNode.valueOf(value)).asText())
                        .isEqualTo(value));

        AgentEventEnvelope streamedFragment = mapper.map(
                new TextBlockDeltaEvent("reply", "content", "/images"));
        assertThat(streamedFragment.payload().path("delta").asText()).isEqualTo("/images");
    }

    @Test
    void stillRedactsSensitiveUnixFileSystemPaths() {
        assertThat(List.of(
                "/etc/passwd",
                "/home/app/.ssh/id_rsa",
                "/Users/admin/.ssh/id_rsa",
                "/workspace/private/secret.txt"))
                .allSatisfy(value -> assertThat(sanitizer.sanitize(TextNode.valueOf(value)).asText())
                        .isNotEqualTo(value));
    }

    @Test
    void replacesBinaryJsonNodesAndRejectsInvalidCreatedAtExplicitly() {
        assertThat(sanitizer.sanitize(BinaryNode.valueOf(new byte[] {1, 2, 3})).asText())
                .isEqualTo("[BINARY_REDACTED]");

        CustomEvent invalid = new CustomEvent(
                "raw-invalid", "not-an-instant", "invalid", Map.of("safe", true));
        assertThatThrownBy(() -> mapper.map(invalid))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("createdAt");
    }

    private List<IdentityCase> identityCases() {
        ToolUseBlock confirmCall = new ToolUseBlock("call-23", "notification_send", Map.of());
        ToolUseBlock externalCall = new ToolUseBlock("call-24", "document_parse", Map.of());
        Msg result = Msg.builder()
                .name("assistant")
                .role(MsgRole.ASSISTANT)
                .textContent("完成")
                .build();

        return List.of(
                identity(new AgentStartEvent("session-1", "reply-1", "assistant"),
                        "reply-1", null, null),
                identity(new AgentEndEvent("reply-2"), "reply-2", null, null),
                identity(new AgentResultEvent(result), null, null, null),
                identity(new ModelCallStartEvent("reply-4"), "reply-4", null, null),
                identity(new ModelCallEndEvent("reply-5", new ChatUsage(1, 2, 0.1)),
                        "reply-5", null, null),
                identity(new TextBlockStartEvent("reply-6", "block-6"),
                        "reply-6", "block-6", null),
                identity(new TextBlockDeltaEvent("reply-7", "block-7", "delta"),
                        "reply-7", "block-7", null),
                identity(new TextBlockEndEvent("reply-8", "block-8"),
                        "reply-8", "block-8", null),
                identity(new ThinkingBlockStartEvent("reply-9", "block-9"),
                        "reply-9", "block-9", null),
                identity(new ThinkingBlockDeltaEvent("reply-10", "block-10", "thought"),
                        "reply-10", "block-10", null),
                identity(new ThinkingBlockEndEvent("reply-11", "block-11"),
                        "reply-11", "block-11", null),
                identity(new DataBlockStartEvent("reply-12", "block-12"),
                        "reply-12", "block-12", null),
                identity(new DataBlockDeltaEvent("reply-13", "block-13", "{}"),
                        "reply-13", "block-13", null),
                identity(new DataBlockEndEvent("reply-14", "block-14"),
                        "reply-14", "block-14", null),
                identity(new ToolCallStartEvent("reply-15", "call-15", "memo_create"),
                        "reply-15", null, "call-15"),
                identity(new ToolCallDeltaEvent(
                        "reply-16", "call-16", "memo_create", "{}"),
                        "reply-16", null, "call-16"),
                identity(new ToolCallEndEvent("reply-17", "call-17", "memo_create"),
                        "reply-17", null, "call-17"),
                identity(new ToolResultStartEvent("reply-18", "call-18", "memo_create"),
                        "reply-18", null, "call-18"),
                identity(new ToolResultTextDeltaEvent(
                        "reply-19", "call-19", "memo_create", "ok"),
                        "reply-19", null, "call-19"),
                identity(new ToolResultDataDeltaEvent(
                        "reply-20", "call-20", "document_parse", null),
                        "reply-20", null, "call-20"),
                identity(new ToolResultEndEvent(
                        "reply-21", "call-21", "memo_create", ToolResultState.SUCCESS),
                        "reply-21", null, "call-21"),
                identity(new ExceedMaxItersEvent("reply-22", 10, 10),
                        "reply-22", null, null),
                identity(new RequireUserConfirmEvent("reply-23", List.of(confirmCall)),
                        "reply-23", null, null),
                identity(new RequireExternalExecutionEvent("reply-24", List.of(externalCall)),
                        "reply-24", null, null),
                identity(new UserConfirmResultEvent(
                        "reply-25", List.of(new ConfirmResult(true, confirmCall))),
                        "reply-25", null, null),
                identity(new ExternalExecutionResultEvent(
                        "reply-26",
                        List.of(new ToolResultBlock("call-24", "document_parse", List.of()))),
                        "reply-26", null, null),
                identity(new RequestStopEvent("全部拒绝", GenerateReason.ALL_TOOLS_DENIED),
                        null, null, null),
                identity(new SubagentExposedEvent(
                        "subagent-28", "agent-28", "session-28", "研究助手"),
                        null, null, null),
                identity(new HintBlockEvent("reply-29", "block-29", "runtime", "hint"),
                        "reply-29", "block-29", null),
                identity(new AllToolsDeniedEvent(List.of(confirmCall)), null, null, null),
                identity(new CustomEvent("state_updated", Map.of("status", "changed")),
                        null, null, null));
    }

    private IdentityCase identity(
            AgentEvent event,
            String replyId,
            String blockId,
            String toolCallId) {
        return new IdentityCase(event, replyId, blockId, toolCallId);
    }

    private record IdentityCase(
            AgentEvent event,
            String replyId,
            String blockId,
            String toolCallId) {
    }
}
