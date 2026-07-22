package com.stonewu.fusion.service.ai.agentscope.tool;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Normalizes platform tool schemas to the strict AgentScope V2 contract. */
public final class AgentScopeToolSchema {

    private static final TypeReference<LinkedHashMap<String, Object>> OBJECT_SCHEMA =
            new TypeReference<>() { };

    private AgentScopeToolSchema() {
    }

    public static PreparedSchema prepare(ObjectMapper objectMapper, String schema, String toolName) {
        Objects.requireNonNull(objectMapper, "objectMapper must not be null");
        requireText(toolName, "toolName");
        if (schema == null || schema.isBlank()) {
            throw new IllegalArgumentException("AgentScope tool schema must not be blank: " + toolName);
        }
        try {
            LinkedHashMap<String, Object> parsed = objectMapper.readValue(schema, OBJECT_SCHEMA);
            if (!"object".equals(parsed.get("type"))) {
                throw new IllegalArgumentException(
                        "AgentScope tool schema root type must be object: " + toolName);
            }
            Object additionalProperties = parsed.get("additionalProperties");
            if (Boolean.TRUE.equals(additionalProperties)) {
                throw new IllegalArgumentException(
                        "AgentScope tool schema must reject additional properties: " + toolName);
            }
            parsed.put("additionalProperties", false);
            parsed.computeIfAbsent("properties", ignored -> Map.of());
            return new PreparedSchema(Map.copyOf(parsed), objectMapper.writeValueAsString(parsed));
        } catch (JsonProcessingException invalidSchema) {
            throw new IllegalArgumentException(
                    "AgentScope tool schema is invalid JSON: " + toolName, invalidSchema);
        }
    }

    public static PreparedSchema prepareSubAgent(
            ObjectMapper objectMapper, String schema, String toolName) {
        return prepare(objectMapper, schema, toolName);
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }

    public record PreparedSchema(Map<String, Object> value, String canonicalJson) {
        public PreparedSchema {
            value = Map.copyOf(Objects.requireNonNull(value, "value must not be null"));
            canonicalJson = requireText(canonicalJson, "canonicalJson");
        }
    }
}
