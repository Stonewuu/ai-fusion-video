package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.config.ai.AiAgentDefinition;
import com.stonewu.fusion.controller.ai.vo.AiChatReqVO;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AiAgentService;
import com.stonewu.fusion.service.ai.AiToolConfigService;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.agentscope.AgentScopeModelFactory;
import com.stonewu.fusion.service.ai.agentscope.context.ProjectContext;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentScopeMcpRegistry;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Creates immutable root and platform-child Harness kernel specifications. */
@Component
public final class AgentKernelSpecFactory {

    public static final String DEFAULT_AGENT_KEY = "ai_assistant_agent";
    public static final String TOOL_WHITELIST_VERSION = "afv-tools-v2";
    private final AiAgentService agentService;
    private final AiToolConfigService toolConfigService;
    private final AgentScopeModelFactory modelFactory;
    private final AgentScopeV2Properties properties;
    private final ObjectMapper objectMapper;
    private final AgentScopeMcpRegistry mcpRegistry;

    @Autowired
    public AgentKernelSpecFactory(
            AiAgentService agentService,
            AiToolConfigService toolConfigService,
            AgentScopeModelFactory modelFactory,
            AgentScopeV2Properties properties,
            ObjectMapper objectMapper,
            AgentScopeMcpRegistry mcpRegistry) {
        this.agentService = Objects.requireNonNull(agentService, "agentService must not be null");
        this.toolConfigService = Objects.requireNonNull(
                toolConfigService, "toolConfigService must not be null");
        this.modelFactory = Objects.requireNonNull(modelFactory, "modelFactory must not be null");
        this.properties = Objects.requireNonNull(properties, "properties must not be null");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
        this.mcpRegistry = Objects.requireNonNull(mcpRegistry, "mcpRegistry must not be null");
    }

    public AgentKernelSpecFactory(
            AiAgentService agentService,
            AiToolConfigService toolConfigService,
            AgentScopeModelFactory modelFactory,
            AgentScopeV2Properties properties,
            ObjectMapper objectMapper) {
        this(
                agentService,
                toolConfigService,
                modelFactory,
                properties,
                objectMapper,
                new AgentScopeMcpRegistry(properties, objectMapper));
    }

    public AgentKernelSpec createRoot(AiChatReqVO request, AiModel model, String systemPrompt) {
        Objects.requireNonNull(request, "request must not be null");
        String requestedType = normalize(request.getAgentType());
        String definitionKey = requestedType == null ? DEFAULT_AGENT_KEY : requestedType;
        AiAgentDefinition definition = requestedType == null
                ? null
                : agentService.getRequiredByType(requestedType);
        String agentName = definition == null
                ? DEFAULT_AGENT_KEY
                : requireText(definition.getName(), "agent.name");
        ToolSelection tools = selectTools(
                requestedType,
                request.getEnabledTools(),
                request.getEnabledMcpTools());
        Map<String, String> promptVariables = AgentPromptVariables.fromRequest(request);
        return create(
                model,
                definitionKey,
                agentName,
                "AgentScope Harness kernel for " + agentName,
                systemPrompt,
                promptVariables,
                tools,
                modelFactory.modelConfigFingerprint(model));
    }

    public AgentKernelSpec createChild(
            AgentKernelSpec parent,
            AiAgentDefinition.SubAgentToolDef subAgent,
            ProjectContext project,
            Map<String, Object> input) {
        Objects.requireNonNull(parent, "parent must not be null");
        Objects.requireNonNull(subAgent, "subAgent must not be null");
        String childType = requireText(subAgent.getRefAgentType(), "subAgent.refAgentType");
        AiAgentDefinition definition = agentService.getRequiredByType(childType);
        String prompt = normalize(subAgent.getSystemPromptOverride());
        if (prompt == null) {
            prompt = requireText(definition.getSystemPrompt(), "subAgent.systemPrompt");
        }
        String instruction = normalize(subAgent.getInstructionOverride());
        if (instruction == null) {
            instruction = normalize(definition.getInstructionTemplate());
        }
        Map<String, String> variables = AgentPromptVariables.forChild(
                parent.promptVariables(), project, input);
        prompt = AgentPromptVariables.render(prompt, variables);
        if (instruction != null) {
            prompt = prompt + "\n\n" + AgentPromptVariables.render(instruction, variables);
        }
        ToolSelection tools = selectTools(childType, null, null);
        String agentName = normalize(subAgent.getToolName());
        if (agentName == null) {
            agentName = requireText(definition.getName(), "subAgent.name");
        }
        return create(
                parent.model(),
                childType,
                agentName,
                requireText(subAgent.getDescription(), "subAgent.description"),
                prompt,
                variables,
                tools,
                parent.key().modelConfigFingerprint());
    }

    private AgentKernelSpec create(
            AiModel model,
            String definitionKey,
            String agentName,
            String description,
            String systemPrompt,
            Map<String, String> promptVariables,
            ToolSelection tools,
            String modelFingerprint) {
        String safePrompt = requireText(systemPrompt, "systemPrompt");
        AgentKernelKey key = AgentKernelKey.create(
                definitionKey,
                modelFingerprint,
                AgentKernelKey.promptVersion(safePrompt, promptVariables),
                tools.manifest(),
                TOOL_WHITELIST_VERSION);
        return new AgentKernelSpec(
                key,
                model,
                definitionKey,
                agentName,
                description,
                safePrompt,
                promptVariables,
                properties.getExecution().getMaxIters(),
                tools.manifest(),
                tools.whitelist(),
                TOOL_WHITELIST_VERSION);
    }

    private ToolSelection selectTools(
            String agentType,
            List<String> requestedTools,
            List<String> requestedMcpTools) {
        Set<String> requested = requestedTools == null || requestedTools.isEmpty()
                ? null
                : Set.copyOf(requestedTools);
        Set<String> requestedMcp = requestedMcpTools == null
                ? null
                : Set.copyOf(requestedMcpTools);
        AiAgentDefinition definition = agentType == null ? null : agentService.getRequiredByType(agentType);
        if (definition != null && !Integer.valueOf(1).equals(definition.getEnableTools())) {
            if (requested != null || requestedMcp != null) {
                throw new IllegalArgumentException("Agent does not enable the requested tools");
            }
            return ToolSelection.empty();
        }
        List<AgentKernelToolManifest> manifest = new ArrayList<>();
        Set<String> whitelist = new LinkedHashSet<>();
        List<ToolExecutor> directTools = agentType == null
                ? toolConfigService.getEnabledTools()
                : toolConfigService.getEnabledToolsByAgent(agentType);
        for (ToolExecutor tool : directTools) {
            if (requested != null && !requested.contains(tool.getToolName())) {
                continue;
            }
            AgentScopeToolSchema.PreparedSchema schema = AgentScopeToolSchema.prepare(
                    objectMapper, tool.getParametersSchema(), tool.getToolName());
            add(manifest, whitelist, new AgentKernelToolManifest(
                    tool.getToolName(),
                    AgentKernelToolManifest.schemaSha256(schema.canonicalJson()),
                    tool.isReadOnly(),
                    tool.isConcurrencySafe()));
        }
        List<AiAgentDefinition.SubAgentToolDef> subAgents = agentType == null
                ? List.of()
                : toolConfigService.getSubAgentTools(agentType);
        for (AiAgentDefinition.SubAgentToolDef subAgent : subAgents) {
            if (requested != null && !requested.contains(subAgent.getToolName())) {
                continue;
            }
            AgentScopeToolSchema.PreparedSchema schema = AgentScopeToolSchema.prepareSubAgent(
                    objectMapper, subAgent.getParametersSchema(), subAgent.getToolName());
            add(manifest, whitelist, new AgentKernelToolManifest(
                    subAgent.getToolName(),
                    AgentKernelToolManifest.schemaSha256(schema.canonicalJson()),
                    false,
                    true));
        }
        if (requested != null && !whitelist.equals(requested)) {
            Set<String> unavailable = new LinkedHashSet<>(requested);
            unavailable.removeAll(whitelist);
            throw new IllegalArgumentException(
                    "Requested AgentScope tools are unavailable: " + unavailable);
        }
        String definitionKey = agentType == null ? DEFAULT_AGENT_KEY : agentType;
        List<AgentKernelToolManifest> availableMcpTools =
                mcpRegistry.manifestsForAgent(definitionKey);
        if (requestedMcp != null) {
            Set<String> availableMcpNames = new LinkedHashSet<>();
            availableMcpTools.forEach(tool -> availableMcpNames.add(tool.toolName()));
            Set<String> unavailableMcp = new LinkedHashSet<>(requestedMcp);
            unavailableMcp.removeAll(availableMcpNames);
            if (!unavailableMcp.isEmpty()) {
                throw new IllegalArgumentException(
                        "Requested AgentScope MCP tools are unavailable: " + unavailableMcp);
            }
        }
        for (AgentKernelToolManifest mcpTool : availableMcpTools) {
            if (requestedMcp == null || requestedMcp.contains(mcpTool.toolName())) {
                add(manifest, whitelist, mcpTool);
            }
        }
        return new ToolSelection(manifest, whitelist);
    }

    private void add(
            List<AgentKernelToolManifest> manifest,
            Set<String> whitelist,
            AgentKernelToolManifest entry) {
        if (!whitelist.add(entry.toolName())) {
            throw new IllegalArgumentException("Duplicate AgentScope tool name: " + entry.toolName());
        }
        manifest.add(entry);
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }

    private static String requireText(String value, String field) {
        String normalized = normalize(value);
        if (normalized == null) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return normalized;
    }

    private record ToolSelection(
            List<AgentKernelToolManifest> manifest,
            Set<String> whitelist) {
        private ToolSelection {
            manifest = List.copyOf(manifest);
            whitelist = Set.copyOf(whitelist);
        }

        private static ToolSelection empty() {
            return new ToolSelection(List.of(), Set.of());
        }
    }
}
