package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.stonewu.fusion.entity.ai.AiModel;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

public record AgentKernelSpec(
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

    public AgentKernelSpec {
        key = Objects.requireNonNull(key, "key must not be null");
        model = copyModel(Objects.requireNonNull(model, "model must not be null"));
        agentDefinitionStableKey = requireText(agentDefinitionStableKey, "agentDefinitionStableKey");
        agentName = requireText(agentName, "agentName");
        description = requireText(description, "description");
        systemPrompt = requireText(systemPrompt, "systemPrompt");
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
        if (!key.promptVersion().equals(AgentKernelKey.promptVersion(systemPrompt))) {
            throw new IllegalArgumentException("AgentKernelSpec key does not match prompt content");
        }
    }

    private static String requireText(String value, String field) {
        return AgentKernelToolManifest.requireText(value, field);
    }

    @Override
    public AiModel model() {
        return copyModel(model);
    }

    private static AiModel copyModel(AiModel source) {
        return AiModel.builder()
                .id(source.getId())
                .name(source.getName())
                .code(source.getCode())
                .modelFamily(source.getModelFamily())
                .modelProtocol(source.getModelProtocol())
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
                .supportReasoning(source.getSupportReasoning())
                .contextWindow(source.getContextWindow())
                .deletedId(source.getDeletedId())
                .build();
    }
}
