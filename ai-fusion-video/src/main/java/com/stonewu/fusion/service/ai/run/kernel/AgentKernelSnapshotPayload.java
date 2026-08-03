package com.stonewu.fusion.service.ai.run.kernel;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import com.stonewu.fusion.service.ai.agentscope.kernel.AgentPromptVariables;

public record AgentKernelSnapshotPayload(
        int schemaVersion,
        String agentDefinitionStableKey,
        String agentName,
        String description,
        String systemPrompt,
        Map<String, String> promptVariables,
        int maxIters,
        String modelConfigId,
        long modelConfigVersion,
        String provider,
        String modelCode,
        JsonNode modelOptions,
        List<ToolManifestSnapshot> tools,
        String applicationVersion) {

    public static final int CURRENT_SCHEMA_VERSION = 2;

    public AgentKernelSnapshotPayload {
        if (schemaVersion != CURRENT_SCHEMA_VERSION) {
            throw new IllegalArgumentException("schemaVersion must be 2");
        }
        agentDefinitionStableKey = ToolManifestSnapshot.requireText(
                agentDefinitionStableKey, "agentDefinitionStableKey");
        agentName = ToolManifestSnapshot.requireText(agentName, "agentName");
        description = ToolManifestSnapshot.requireText(description, "description");
        systemPrompt = ToolManifestSnapshot.requireText(systemPrompt, "systemPrompt");
        promptVariables = AgentPromptVariables.immutable(promptVariables);
        if (maxIters <= 0) {
            throw new IllegalArgumentException("maxIters must be greater than zero");
        }
        modelConfigId = ToolManifestSnapshot.requireText(modelConfigId, "modelConfigId");
        if (modelConfigVersion <= 0) {
            throw new IllegalArgumentException("modelConfigVersion must be positive");
        }
        provider = ToolManifestSnapshot.requireText(provider, "provider");
        modelCode = ToolManifestSnapshot.requireText(modelCode, "modelCode");
        modelOptions = Objects.requireNonNull(modelOptions, "modelOptions must not be null")
                .deepCopy();
        if (!modelOptions.isObject()) {
            throw new IllegalArgumentException("modelOptions must be a JSON object");
        }
        tools = List.copyOf(Objects.requireNonNull(tools, "tools must not be null"));
        Set<String> names = new HashSet<>();
        for (ToolManifestSnapshot tool : tools) {
            Objects.requireNonNull(tool, "tools entry must not be null");
            if (!names.add(tool.name())) {
                throw new IllegalArgumentException("duplicate tool snapshot: " + tool.name());
            }
        }
        applicationVersion = ToolManifestSnapshot.requireText(
                applicationVersion, "applicationVersion");
    }

    public AgentKernelSnapshotPayload(
            int schemaVersion,
            String agentDefinitionStableKey,
            String agentName,
            String description,
            String systemPrompt,
            int maxIters,
            String modelConfigId,
            long modelConfigVersion,
            String provider,
            String modelCode,
            JsonNode modelOptions,
            List<ToolManifestSnapshot> tools,
            String applicationVersion) {
        this(schemaVersion, agentDefinitionStableKey, agentName, description, systemPrompt,
                Map.of(), maxIters, modelConfigId, modelConfigVersion, provider, modelCode,
                modelOptions, tools, applicationVersion);
    }

    @Override
    public JsonNode modelOptions() {
        return modelOptions.deepCopy();
    }
}
