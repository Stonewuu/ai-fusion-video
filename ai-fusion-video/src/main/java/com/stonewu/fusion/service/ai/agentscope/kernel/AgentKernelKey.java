package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.stonewu.fusion.entity.ai.AiModel;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

public record AgentKernelKey(
        String agentDefinitionStableKey,
        String modelConfigFingerprint,
        String promptVersion,
        String toolManifestFingerprint,
        String toolWhitelistVersion) {

    public AgentKernelKey {
        agentDefinitionStableKey = requireText(agentDefinitionStableKey, "agentDefinitionStableKey");
        modelConfigFingerprint = requireText(modelConfigFingerprint, "modelConfigFingerprint");
        promptVersion = requireText(promptVersion, "promptVersion");
        toolManifestFingerprint = requireText(toolManifestFingerprint, "toolManifestFingerprint");
        toolWhitelistVersion = requireText(toolWhitelistVersion, "toolWhitelistVersion");
    }

    public static AgentKernelKey create(
            String agentDefinitionStableKey,
            String modelConfigFingerprint,
            String promptVersion,
            Collection<AgentKernelToolManifest> toolManifest,
            String toolWhitelistVersion) {
        return new AgentKernelKey(
                agentDefinitionStableKey,
                modelConfigFingerprint,
                promptVersion,
                manifestFingerprint(toolManifest),
                toolWhitelistVersion);
    }

    public static String manifestFingerprint(Collection<AgentKernelToolManifest> manifest) {
        List<AgentKernelToolManifest> sorted = new ArrayList<>(
                java.util.Objects.requireNonNull(manifest, "manifest must not be null"));
        sorted.sort(Comparator.comparing(AgentKernelToolManifest::toolName));
        HashSet<String> names = new HashSet<>();
        StringBuilder canonical = new StringBuilder();
        for (AgentKernelToolManifest entry : sorted) {
            java.util.Objects.requireNonNull(entry, "manifest entry must not be null");
            if (!names.add(entry.toolName())) {
                throw new IllegalArgumentException("duplicate tool manifest entry: " + entry.toolName());
            }
            if (!canonical.isEmpty()) {
                canonical.append('\n');
            }
            canonical.append(entry.toolName())
                    .append('|').append(entry.schemaSha256())
                    .append('|').append(entry.readOnly())
                    .append('|').append(entry.concurrencySafe());
        }
        return AgentKernelToolManifest.sha256(canonical.toString());
    }

    public static String whitelistVersion(String schemaVersion, Collection<String> whitelist) {
        String normalizedVersion = requireText(schemaVersion, "schemaVersion");
        List<String> sorted = new ArrayList<>(
                java.util.Objects.requireNonNull(whitelist, "whitelist must not be null"));
        sorted.replaceAll(value -> requireText(value, "whitelist tool name"));
        sorted.sort(String::compareTo);
        if (new HashSet<>(sorted).size() != sorted.size()) {
            throw new IllegalArgumentException("whitelist must not contain duplicate tool names");
        }
        StringBuilder canonical = new StringBuilder();
        appendField(canonical, normalizedVersion);
        sorted.forEach(value -> appendField(canonical, value));
        return normalizedVersion + ":" + AgentKernelToolManifest.sha256(canonical.toString());
    }

    public static String contentFingerprint(String namespace, Object... values) {
        StringBuilder canonical = new StringBuilder();
        appendField(canonical, requireText(namespace, "namespace"));
        for (Object value : values) {
            appendField(canonical, value == null ? "<null>" : value.toString());
        }
        return AgentKernelToolManifest.sha256(canonical.toString());
    }

    public static String modelRecordFingerprint(AiModel model) {
        java.util.Objects.requireNonNull(model, "model must not be null");
        return contentFingerprint(
                "ai-model-v1",
                model.getId(),
                model.getCode(),
                model.getModelProtocol(),
                model.getCapabilityPresetCode(),
                model.getConfig(),
                model.getApiConfigId(),
                model.getMaxConcurrency(),
                model.getSupportVision(),
                model.getSupportReasoning(),
                model.getReasoningEffortLevels(),
                model.getContextWindow());
    }

    public static String promptVersion(String systemPrompt) {
        return contentFingerprint(
                "agent-prompt-v1",
                requireText(systemPrompt, "systemPrompt"));
    }

    public static String promptVersion(
            String systemPrompt, Map<String, String> promptVariables) {
        Map<String, String> safeVariables = AgentPromptVariables.immutable(promptVariables);
        if (safeVariables.isEmpty()) {
            return promptVersion(systemPrompt);
        }
        return contentFingerprint(
                "agent-prompt-v2",
                requireText(systemPrompt, "systemPrompt"),
                AgentPromptVariables.canonical(safeVariables));
    }

    private static void appendField(StringBuilder target, String value) {
        target.append(value.length()).append(':').append(value).append('\n');
    }

    private static String requireText(String value, String field) {
        return AgentKernelToolManifest.requireText(value, field);
    }
}
