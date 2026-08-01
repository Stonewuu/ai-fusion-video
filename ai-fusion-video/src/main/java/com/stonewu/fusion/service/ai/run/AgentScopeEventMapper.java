package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import io.agentscope.core.event.AgentEndEvent;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.event.AgentResultEvent;
import io.agentscope.core.event.AgentStartEvent;
import io.agentscope.core.event.DataBlockDeltaEvent;
import io.agentscope.core.event.DataBlockEndEvent;
import io.agentscope.core.event.DataBlockStartEvent;
import io.agentscope.core.event.ExceedMaxItersEvent;
import io.agentscope.core.event.ExternalExecutionResultEvent;
import io.agentscope.core.event.HintBlockEvent;
import io.agentscope.core.event.ModelCallEndEvent;
import io.agentscope.core.event.ModelCallStartEvent;
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
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.Map;
import java.util.Objects;

@Component
public final class AgentScopeEventMapper {

    private static final String MAIN_SOURCE = "main";
    private static final String PARENT_TOOL_CALL_ID = "parentToolCallId";
    private static final String PARENT_TOOL_CALL_ID_SNAKE = "parent_tool_call_id";
    private static final String AGENT_NAME = "agentName";
    private static final String AGENT_NAME_SNAKE = "agent_name";

    private final ObjectMapper objectMapper;
    private final AgentEventEnvelopeSanitizer sanitizer;

    public AgentScopeEventMapper(ObjectMapper objectMapper) {
        this(objectMapper, new AgentEventEnvelopeSanitizer());
    }

    @Autowired
    public AgentScopeEventMapper(
            ObjectMapper objectMapper,
            AgentEventEnvelopeSanitizer sanitizer) {
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
        this.sanitizer = Objects.requireNonNull(sanitizer, "sanitizer must not be null");
    }

    public AgentEventEnvelope map(AgentEvent event) {
        Objects.requireNonNull(event, "event must not be null");
        Objects.requireNonNull(event.getType(), "event.type must not be null");
        Identity identity = identity(event);
        JsonNode payload = sanitizer.sanitize(objectMapper.valueToTree(event));
        return new AgentEventEnvelope(
                requireText(event.getId(), "event.id"),
                event.getType().getValue(),
                optionalText(event.getSource()),
                optionalText(identity.replyId()),
                optionalText(identity.blockId()),
                optionalText(identity.toolCallId()),
                optionalText(identity.parentToolCallId()),
                optionalText(identity.agentName()),
                outputType(event),
                payload,
                createdAt(event));
    }

    private Identity identity(AgentEvent event) {
        Identity base = baseIdentity(event);
        return switch (event.getType()) {
            case AGENT_START -> base.withReplyAndAgent(
                    ((AgentStartEvent) event).getReplyId(),
                    ((AgentStartEvent) event).getName());
            case AGENT_END -> {
                AgentEndEvent agentEnd = (AgentEndEvent) event;
                yield isMainSource(event.getSource())
                        ? new Identity(agentEnd.getReplyId(), null, null, null, null)
                        : base.withReply(agentEnd.getReplyId());
            }
            case AGENT_RESULT -> {
                AgentResultEvent result = (AgentResultEvent) event;
                String resultAgentName = result.getResult() != null
                        ? result.getResult().getName()
                        : null;
                yield isMainSource(event.getSource())
                        ? new Identity(null, null, null, null, null)
                        : base.withAgent(
                                base.agentName() != null ? base.agentName() : resultAgentName);
            }
            case MODEL_CALL_START -> base.withReply(((ModelCallStartEvent) event).getReplyId());
            case MODEL_CALL_END -> base.withReply(((ModelCallEndEvent) event).getReplyId());
            case TEXT_BLOCK_START -> base.withReplyAndBlock(
                    ((TextBlockStartEvent) event).getReplyId(),
                    ((TextBlockStartEvent) event).getBlockId());
            case TEXT_BLOCK_DELTA -> base.withReplyAndBlock(
                    ((TextBlockDeltaEvent) event).getReplyId(),
                    ((TextBlockDeltaEvent) event).getBlockId());
            case TEXT_BLOCK_END -> base.withReplyAndBlock(
                    ((TextBlockEndEvent) event).getReplyId(),
                    ((TextBlockEndEvent) event).getBlockId());
            case THINKING_BLOCK_START -> base.withReplyAndBlock(
                    ((ThinkingBlockStartEvent) event).getReplyId(),
                    ((ThinkingBlockStartEvent) event).getBlockId());
            case THINKING_BLOCK_DELTA -> base.withReplyAndBlock(
                    ((ThinkingBlockDeltaEvent) event).getReplyId(),
                    ((ThinkingBlockDeltaEvent) event).getBlockId());
            case THINKING_BLOCK_END -> base.withReplyAndBlock(
                    ((ThinkingBlockEndEvent) event).getReplyId(),
                    ((ThinkingBlockEndEvent) event).getBlockId());
            case DATA_BLOCK_START -> base.withReplyAndBlock(
                    ((DataBlockStartEvent) event).getReplyId(),
                    ((DataBlockStartEvent) event).getBlockId());
            case DATA_BLOCK_DELTA -> base.withReplyAndBlock(
                    ((DataBlockDeltaEvent) event).getReplyId(),
                    ((DataBlockDeltaEvent) event).getBlockId());
            case DATA_BLOCK_END -> base.withReplyAndBlock(
                    ((DataBlockEndEvent) event).getReplyId(),
                    ((DataBlockEndEvent) event).getBlockId());
            case TOOL_CALL_START -> base.withReplyAndTool(
                    ((ToolCallStartEvent) event).getReplyId(),
                    ((ToolCallStartEvent) event).getToolCallId());
            case TOOL_CALL_DELTA -> base.withReplyAndTool(
                    ((ToolCallDeltaEvent) event).getReplyId(),
                    ((ToolCallDeltaEvent) event).getToolCallId());
            case TOOL_CALL_END -> base.withReplyAndTool(
                    ((ToolCallEndEvent) event).getReplyId(),
                    ((ToolCallEndEvent) event).getToolCallId());
            case TOOL_RESULT_START -> base.withReplyAndTool(
                    ((ToolResultStartEvent) event).getReplyId(),
                    ((ToolResultStartEvent) event).getToolCallId());
            case TOOL_RESULT_TEXT_DELTA -> base.withReplyAndTool(
                    ((ToolResultTextDeltaEvent) event).getReplyId(),
                    ((ToolResultTextDeltaEvent) event).getToolCallId());
            case TOOL_RESULT_DATA_DELTA -> base.withReplyAndTool(
                    ((ToolResultDataDeltaEvent) event).getReplyId(),
                    ((ToolResultDataDeltaEvent) event).getToolCallId());
            case TOOL_RESULT_END -> base.withReplyAndTool(
                    ((ToolResultEndEvent) event).getReplyId(),
                    ((ToolResultEndEvent) event).getToolCallId());
            case EXCEED_MAX_ITERS -> base.withReply(
                    ((ExceedMaxItersEvent) event).getReplyId());
            case REQUIRE_USER_CONFIRM -> base.withReply(
                    ((RequireUserConfirmEvent) event).getReplyId());
            case REQUIRE_EXTERNAL_EXECUTION -> base.withReply(
                    ((RequireExternalExecutionEvent) event).getReplyId());
            case USER_CONFIRM_RESULT -> base.withReply(
                    ((UserConfirmResultEvent) event).getReplyId());
            case EXTERNAL_EXECUTION_RESULT -> base.withReply(
                    ((ExternalExecutionResultEvent) event).getReplyId());
            case REQUEST_STOP -> base;
            case SUBAGENT_EXPOSED -> base.withAgent(
                    ((SubagentExposedEvent) event).getAgentId());
            case HINT_BLOCK -> base.withReplyAndBlock(
                    ((HintBlockEvent) event).getReplyId(),
                    ((HintBlockEvent) event).getBlockId());
            case ALL_TOOLS_DENIED -> base;
            case CUSTOM -> base;
        };
    }

    private String outputType(AgentEvent event) {
        return switch (event.getType()) {
            case TEXT_BLOCK_DELTA -> "CONTENT";
            case THINKING_BLOCK_DELTA -> "REASONING";
            case TOOL_CALL_START -> "TOOL_CALL_STARTED";
            case TOOL_CALL_END -> "TOOL_CALL";
            case TOOL_RESULT_END -> "TOOL_FINISHED";
            case SUBAGENT_EXPOSED -> "SUB_AGENT_STARTED";
            case AGENT_END -> isMainSource(event.getSource()) ? null : "SUB_AGENT_FINISHED";
            // The raw pause event is journaled as an internal checkpoint. The
            // durable waiting-state service publishes the actionable event only
            // after the RUNNING -> WAITING_CONFIRMATION transition commits.
            case REQUIRE_USER_CONFIRM -> null;
            case REQUIRE_EXTERNAL_EXECUTION -> "EXTERNAL_EXECUTION_REQUIRED";
            // The durable waiting-state transition publishes the authoritative
            // decision event before execution resumes. The AgentScope echo is
            // journaled only and must not produce a second frontend decision.
            case USER_CONFIRM_RESULT -> null;
            case EXTERNAL_EXECUTION_RESULT -> "EXTERNAL_EXECUTION_RESULT";
            case AGENT_START,
                    AGENT_RESULT,
                    MODEL_CALL_START,
                    MODEL_CALL_END,
                    TEXT_BLOCK_START,
                    TEXT_BLOCK_END,
                    THINKING_BLOCK_START,
                    THINKING_BLOCK_END,
                    DATA_BLOCK_START,
                    DATA_BLOCK_DELTA,
                    DATA_BLOCK_END,
                    TOOL_CALL_DELTA,
                    TOOL_RESULT_START,
                    TOOL_RESULT_TEXT_DELTA,
                    TOOL_RESULT_DATA_DELTA,
                    EXCEED_MAX_ITERS,
                    REQUEST_STOP,
                    HINT_BLOCK,
                    ALL_TOOLS_DENIED,
                    CUSTOM -> null;
        };
    }

    private Identity baseIdentity(AgentEvent event) {
        String sourceAgent = agentNameFromSource(event.getSource());
        Map<String, Object> metadata = event.getMetadata();
        if (metadata == null || metadata.isEmpty()) {
            return new Identity(null, null, null, null, sourceAgent);
        }
        String parentToolCallId = firstMetadataText(
                metadata, PARENT_TOOL_CALL_ID, PARENT_TOOL_CALL_ID_SNAKE);
        String metadataAgentName = firstMetadataText(metadata, AGENT_NAME, AGENT_NAME_SNAKE);
        return new Identity(
                null,
                null,
                null,
                parentToolCallId,
                metadataAgentName != null ? metadataAgentName : sourceAgent);
    }

    private String firstMetadataText(Map<String, Object> metadata, String... keys) {
        for (String key : keys) {
            if (!metadata.containsKey(key)) {
                continue;
            }
            Object value = metadata.get(key);
            if (value == null) {
                return null;
            }
            if (!(value instanceof String text)) {
                throw new IllegalArgumentException("Agent event identity metadata must be text: " + key);
            }
            return requireText(text, "event.metadata." + key);
        }
        return null;
    }

    private String agentNameFromSource(String source) {
        if (isMainSource(source)) {
            return null;
        }
        int separator = source.lastIndexOf('/');
        String name = separator >= 0 ? source.substring(separator + 1) : source;
        return requireText(name, "event.source agent name");
    }

    private boolean isMainSource(String source) {
        return source == null || source.isBlank() || MAIN_SOURCE.equals(source);
    }

    private Instant createdAt(AgentEvent event) {
        try {
            return Instant.parse(requireText(event.getCreatedAt(), "event.createdAt"));
        } catch (DateTimeParseException invalidCreatedAt) {
            throw new IllegalArgumentException("Agent event createdAt must be an ISO-8601 instant");
        }
    }

    private static String requireText(String value, String field) {
        Objects.requireNonNull(value, field + " must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }

    private static String optionalText(String value) {
        return value == null || value.isBlank() ? null : value;
    }

    private record Identity(
            String replyId,
            String blockId,
            String toolCallId,
            String parentToolCallId,
            String agentName) {

        private Identity withReply(String value) {
            return new Identity(value, blockId, toolCallId, parentToolCallId, agentName);
        }

        private Identity withAgent(String value) {
            return new Identity(
                    replyId,
                    blockId,
                    toolCallId,
                    parentToolCallId,
                    optionalText(value) != null ? value : agentName);
        }

        private Identity withReplyAndAgent(String reply, String agent) {
            return withReply(reply).withAgent(agent);
        }

        private Identity withReplyAndBlock(String reply, String block) {
            return new Identity(reply, block, null, parentToolCallId, agentName);
        }

        private Identity withReplyAndTool(String reply, String tool) {
            return new Identity(reply, null, tool, parentToolCallId, agentName);
        }
    }
}
