package com.stonewu.fusion.service.ai.run;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/** Assigns immutable AgentState identities to conversation and execution generations. */
public final class AgentStateSessionIds {

    private static final String CONVERSATION_PREFIX = "afv:v2:";
    private static final String ROOT_GENERATION_PREFIX = "afv-root:";
    private static final String CHILD_GENERATION_PREFIX = "afv-child:";

    private AgentStateSessionIds() {
    }

    public static String conversation(String conversationId, String agentDefinitionStableKey) {
        return CONVERSATION_PREFIX
                + requireComponent(conversationId, "conversationId")
                + ':'
                + requireComponent(agentDefinitionStableKey, "agentDefinitionStableKey");
    }

    public static String recoveryGeneration(
            String conversationId,
            String agentDefinitionStableKey,
            String runId) {
        return hashed(
                ROOT_GENERATION_PREFIX,
                requireComponent(conversationId, "conversationId"),
                requireComponent(agentDefinitionStableKey, "agentDefinitionStableKey"),
                requireComponent(runId, "runId"));
    }

    public static String childGeneration(
            String parentRunId,
            String parentStateSessionId,
            String parentToolCallId,
            String agentDefinitionStableKey) {
        return hashed(
                CHILD_GENERATION_PREFIX,
                requireComponent(parentRunId, "parentRunId"),
                requireComponent(parentStateSessionId, "parentStateSessionId"),
                requireComponent(parentToolCallId, "parentToolCallId"),
                requireComponent(agentDefinitionStableKey, "agentDefinitionStableKey"));
    }

    private static String hashed(String prefix, String... components) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            for (String component : components) {
                digest.update(component.getBytes(StandardCharsets.UTF_8));
                digest.update((byte) 0);
            }
            return prefix + HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private static String requireComponent(String value, String field) {
        if (value == null || value.isBlank() || value.indexOf('\0') >= 0) {
            throw new IllegalArgumentException(field + " must be a non-blank session component");
        }
        return value;
    }
}
