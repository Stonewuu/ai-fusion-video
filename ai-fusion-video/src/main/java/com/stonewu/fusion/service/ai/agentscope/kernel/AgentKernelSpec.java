package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.ToolManifestSnapshot;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public record AgentKernelSpec(
        AgentKernelKey key,
        AiModel model,
        String agentDefinitionStableKey,
        String agentName,
        String description,
        String systemPrompt,
        Map<String, String> promptVariables,
        int maxIters,
        List<AgentKernelToolManifest> toolManifest,
        Set<String> toolWhitelist,
        String toolWhitelistVersion) {

    public AgentKernelSpec {
        key = Objects.requireNonNull(key, "key must not be null");
        model = copyModel(Objects.requireNonNull(model, "model must not be null"));
        agentDefinitionStableKey = requireText(agentDefinitionStableKey, "agentDefinitionStableKey");
        agentName = requireText(agentName, "agentName");
        description = requireText(description, "description");
        systemPrompt = requireText(systemPrompt, "systemPrompt");
        promptVariables = AgentPromptVariables.immutable(promptVariables);
        toolWhitelistVersion = requireText(toolWhitelistVersion, "toolWhitelistVersion");
        if (maxIters <= 0) {
            throw new IllegalArgumentException("maxIters must be greater than zero");
        }

        toolManifest = List.copyOf(Objects.requireNonNull(toolManifest, "toolManifest must not be null"));
        toolWhitelist = Set.copyOf(Objects.requireNonNull(toolWhitelist, "toolWhitelist must not be null"));
        Set<String> manifestNames = new HashSet<>();
        for (AgentKernelToolManifest entry : toolManifest) {
            Objects.requireNonNull(entry, "toolManifest entry must not be null");
            if (!manifestNames.add(entry.toolName())) {
                throw new IllegalArgumentException("duplicate tool manifest entry: " + entry.toolName());
            }
        }
        if (!manifestNames.equals(toolWhitelist)) {
            throw new IllegalArgumentException(
                    "toolWhitelist must exactly match the manifest tool names");
        }
        if (!key.agentDefinitionStableKey().equals(agentDefinitionStableKey)) {
            throw new IllegalArgumentException("AgentKernelSpec key does not match agent definition");
        }
        if (!key.toolWhitelistVersion().equals(toolWhitelistVersion)) {
            throw new IllegalArgumentException("AgentKernelSpec key does not match whitelist version");
        }
        if (!key.toolManifestFingerprint().equals(AgentKernelKey.manifestFingerprint(toolManifest))) {
            throw new IllegalArgumentException("AgentKernelSpec key does not match tool manifest");
        }
        if (!key.promptVersion().equals(
                AgentKernelKey.promptVersion(systemPrompt, promptVariables))) {
            throw new IllegalArgumentException("AgentKernelSpec key does not match prompt content");
        }
    }

    public AgentKernelSpec(
            AgentKernelKey key,
            AiModel model,
            String agentDefinitionStableKey,
            String agentName,
            String description,
            String systemPrompt,
            int maxIters,
            List<AgentKernelToolManifest> toolManifest,
            Set<String> toolWhitelist,
            String toolWhitelistVersion) {
        this(key, model, agentDefinitionStableKey, agentName, description, systemPrompt,
                Map.of(), maxIters, toolManifest, toolWhitelist, toolWhitelistVersion);
    }

    private static String requireText(String value, String field) {
        return AgentKernelToolManifest.requireText(value, field);
    }

    @Override
    public AiModel model() {
        return copyModel(model);
    }

    public AgentKernelSnapshotPayload snapshotPayload(
            String modelConfigId,
            long modelConfigVersion,
            String provider,
            JsonNode modelOptions,
            List<ToolManifestSnapshot> tools,
            String applicationVersion) {
        return new AgentKernelSnapshotPayload(
                AgentKernelSnapshotPayload.CURRENT_SCHEMA_VERSION,
                agentDefinitionStableKey,
                agentName,
                description,
                systemPrompt,
                promptVariables,
                maxIters,
                modelConfigId,
                modelConfigVersion,
                provider,
                requireText(model.getCode(), "model.code"),
                modelOptions,
                tools,
                applicationVersion);
    }

    private static AiModel copyModel(AiModel source) {
        return AiModel.builder()
                .id(source.getId())
                .name(source.getName())
                .code(source.getCode())
                .modelProtocol(source.getModelProtocol())
                .capabilityPresetCode(source.getCapabilityPresetCode())
                .modelType(source.getModelType())
                .icon(source.getIcon())
                .description(source.getDescription())
                .sort(source.getSort())
                .status(source.getStatus())
                .config(source.getConfig())
                .maxConcurrency(source.getMaxConcurrency())
                .apiConfigId(source.getApiConfigId())
                .defaultModel(source.getDefaultModel())
                .supportVision(source.getSupportVision())
                .multimodalInputTypes(source.getMultimodalInputTypes())
                .multimodalInputTransports(source.getMultimodalInputTransports())
                .supportReasoning(source.getSupportReasoning())
                .reasoningEffortLevels(source.getReasoningEffortLevels())
                .contextWindow(source.getContextWindow())
                .deletedId(source.getDeletedId())
                .build();
    }
}
