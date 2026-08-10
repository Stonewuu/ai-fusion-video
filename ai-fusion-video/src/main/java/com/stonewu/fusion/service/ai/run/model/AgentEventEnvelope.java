package com.stonewu.fusion.service.ai.run.model;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.time.Instant;
import java.util.Objects;

public record AgentEventEnvelope(
        String rawEventId,
        String rawEventType,
        String source,
        String replyId,
        String blockId,
        String toolCallId,
        String parentToolCallId,
        String agentName,
        String outputType,
        JsonNode payload,
        Instant createdAt) {

    public AgentEventEnvelope {
        rawEventId = requireText(rawEventId, "rawEventId");
        rawEventType = requireText(rawEventType, "rawEventType");
        source = optionalText(source, "source");
        replyId = optionalText(replyId, "replyId");
        blockId = optionalText(blockId, "blockId");
        toolCallId = optionalText(toolCallId, "toolCallId");
        parentToolCallId = optionalText(parentToolCallId, "parentToolCallId");
        agentName = optionalText(agentName, "agentName");
        outputType = optionalText(outputType, "outputType");
        payload = Objects.requireNonNull(payload, "payload must not be null").deepCopy();
        Objects.requireNonNull(createdAt, "createdAt must not be null");
    }

    @Override
    public JsonNode payload() {
        return payload.deepCopy();
    }

    public String payloadJson(ObjectMapper objectMapper) {
        try {
            return Objects.requireNonNull(objectMapper, "objectMapper must not be null")
                    .writeValueAsString(payload);
        } catch (JsonProcessingException serializationFailure) {
            throw new IllegalStateException("Agent event payload serialization failed", serializationFailure);
        }
    }

    private static String requireText(String value, String field) {
        Objects.requireNonNull(value, field + " must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }

    private static String optionalText(String value, String field) {
        if (value != null && value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank when present");
        }
        return value;
    }
}
