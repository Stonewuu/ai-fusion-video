package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.agentscope.core.event.ConfirmResult;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.ToolCallState;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.message.UserMessage;

import java.util.*;

/**
 * HTTP 审批和后台恢复共用完全相同的调用身份及批准/拒绝消息。
 */
public final class AgentConfirmationMessages {
    private static final TypeReference<Map<String, Object>> OBJECT_MAP = new TypeReference<>() {
    };

    private AgentConfirmationMessages() {
    }

    public static Msg restore(String suspended, Map<String, Boolean> decisions, ObjectMapper mapper) {
        try {
            JsonNode calls = mapper.readTree(suspended);
            if (!calls.isArray() || calls.isEmpty() || decisions.isEmpty()) {
                throw invalid();
            }
            List<ConfirmResult> results = new ArrayList<>();
            var identities = new HashSet<String>();
            for (JsonNode value : calls) {
                String id = text(value, "id");
                if (!identities.add(id) || decisions.get(id) == null) {
                    throw invalid();
                }
                var input = object(value.path("input"), mapper);
                var metadata = new LinkedHashMap<>(object(value.path("metadata"), mapper));
                if (metadata.get(ToolUseBlock.METADATA_THOUGHT_SIGNATURE) instanceof String encoded) {
                    metadata.put(ToolUseBlock.METADATA_THOUGHT_SIGNATURE, Base64.getDecoder().decode(encoded));
                }
                var call = new ToolUseBlock(id, text(value, "name"), input,
                        value.hasNonNull("content") ? value.get("content").asText() : null,
                        Map.copyOf(metadata), ToolCallState.valueOf(text(value, "state")));
                results.add(new ConfirmResult(decisions.get(id), call));
            }
            if (!identities.equals(decisions.keySet())) {
                throw invalid();
            }
            return UserMessage.builder().metadata(Map.of(Msg.METADATA_CONFIRM_RESULTS, results)).build();
        } catch (JsonProcessingException | IllegalArgumentException failure) {
            throw new IllegalStateException("持久化的审批消息无效", failure);
        }
    }

    private static Map<String, Object> object(JsonNode value, ObjectMapper mapper) {
        if (!value.isObject()) {
            throw invalid();
        }
        return mapper.convertValue(value, OBJECT_MAP);
    }

    private static String text(JsonNode value, String name) {
        if (!value.path(name).isTextual() || value.path(name).asText().isBlank()) {
            throw invalid();
        }
        return value.path(name).asText();
    }

    private static IllegalStateException invalid() {
        return new IllegalStateException("审批调用身份与决定不一致");
    }
}
