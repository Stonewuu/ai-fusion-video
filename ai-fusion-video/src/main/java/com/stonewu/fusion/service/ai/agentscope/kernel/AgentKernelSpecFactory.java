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
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.configuration.AgentDefinitionConfiguration;
import com.stonewu.fusion.service.ai.configuration.AgentKernelSkills;
import com.stonewu.fusion.service.ai.run.AgentExecutionLimits;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.ToolManifestSnapshot;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentToolNames;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeTypes;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Creates immutable root and platform-child Harness kernel specifications.
 */
@Component
public final class AgentKernelSpecFactory {

    public static final String DEFAULT_AGENT_KEY = "ai_assistant_agent";
    public static final String TOOL_WHITELIST_VERSION = "afv-tools-v2";
    public static final String OWNER_USER_ID_VARIABLE = "agentWorkspaceUserId";
    public static final String MCP_CONFIG_FINGERPRINT_VARIABLE = "agentMcpConfigFingerprint";
    public static final String TOOL_EXECUTION_MODE_VARIABLE = "agentToolExecutionMode";
    private final AiAgentService agentService;
    private final AiToolConfigService toolConfigService;
    private final AgentScopeModelFactory modelFactory;
    private final AgentScopeV2Properties properties;
    private final ObjectMapper objectMapper;
    private final AgentScopeMcpRegistry mcpRegistry;
    private final AgentKernelSkills skills;

    @Autowired
    public AgentKernelSpecFactory(
            AiAgentService agentService,
            AiToolConfigService toolConfigService,
            AgentScopeModelFactory modelFactory,
            AgentScopeV2Properties properties,
            ObjectMapper objectMapper,
            AgentScopeMcpRegistry mcpRegistry,
            AgentKernelSkills skills) {
        this.agentService = Objects.requireNonNull(agentService, "agentService must not be null");
        this.toolConfigService = Objects.requireNonNull(
                toolConfigService, "toolConfigService must not be null");
        this.modelFactory = Objects.requireNonNull(modelFactory, "modelFactory must not be null");
        this.properties = Objects.requireNonNull(properties, "properties must not be null");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
        this.mcpRegistry = Objects.requireNonNull(mcpRegistry, "mcpRegistry must not be null");
        this.skills = Objects.requireNonNull(skills, "skills must not be null");
    }

    public AgentKernelSpecFactory(
            AiAgentService agentService,
            AiToolConfigService toolConfigService,
            AgentScopeModelFactory modelFactory,
            AgentScopeV2Properties properties,
            ObjectMapper objectMapper,
            AgentKernelSkills skills) {
        this(
                agentService,
                toolConfigService,
                modelFactory,
                properties,
                objectMapper,
                new AgentScopeMcpRegistry(properties, objectMapper), skills);
    }

    public AgentKernelSpec createRoot(AiChatReqVO request, AiModel model, String systemPrompt) {
        return createRoot(request, model, systemPrompt, null);
    }

    public AgentKernelSpec createRoot(
            AiChatReqVO request,
            AiModel model,
            String systemPrompt,
            Long ownerUserId) {
        return createRoot(request, model, systemPrompt, ownerUserId, false);
    }

    /**
     * 仅由已受理的专业画布节点使用；HTTP 请求没有开启此工具范围的字段。
     */
    public AgentKernelSpec createCanvasNode(AiChatReqVO request, AiModel model, String systemPrompt, Long ownerUserId) {
        if (!CanvasAgentNodeTypes.TYPES.contains(request.getAgentType()) || request.getCanvasContext() == null) {
            throw new IllegalArgumentException("专业节点缺少可信的智能体类型或画布上下文");
        }
        return createRoot(request, model, systemPrompt, ownerUserId, true);
    }

    private AgentKernelSpec createRoot(AiChatReqVO request, AiModel model, String systemPrompt, Long ownerUserId,
                                       boolean canvasNode) {
        Objects.requireNonNull(request, "request must not be null");
        String requestedType = normalize(request.getAgentType());
        String definitionKey = requestedType == null ? DEFAULT_AGENT_KEY : requestedType;
        Map<String, String> promptVariables = new LinkedHashMap<>(AgentPromptVariables.fromRequest(request));
        String configuration = request.getAgentConfigurationSnapshot() == null
                ? agentService.snapshot(requestedType, ownerUserId) : request.getAgentConfigurationSnapshot();
        promptVariables.put(AgentDefinitionConfiguration.SNAPSHOT_VARIABLE, configuration);
        AiAgentDefinition definition = requestedType == null
                ? null
                : agentService.fromSnapshot(requestedType, promptVariables);
        String agentName = definition == null
                ? DEFAULT_AGENT_KEY
                : requireText(definition.getName(), "agent.name");
        ToolSelection tools = selectTools(
                requestedType,
                request.getEnabledTools(),
                canvasNode && Integer.valueOf(1)
                        .equals(definition.getEnableTools()) ? List.of() : request.getEnabledMcpTools(),
                ownerUserId, promptVariables);
        if (canvasNode) {
            tools = nodeTools(tools);
        } else if (request.getCanvasContext() != null) {
            // 绑定画布的根助手只能使用能验证该作用域的工具，不能借旧项目工具绕开画布边界。
            Set<String> permitted = new LinkedHashSet<>(CanvasAgentToolNames.NAMES);
            if (request.getEnabledMcpTools() != null) {
                permitted.addAll(request.getEnabledMcpTools());
            }
            tools = new ToolSelection(
                    tools.manifest().stream().filter(item -> permitted.contains(item.toolName())).toList(),
                    tools.whitelist().stream().filter(permitted::contains)
                            .collect(Collectors.toCollection(LinkedHashSet::new)));
        } else {
            tools = new ToolSelection(
                    tools.manifest().stream().filter(item -> !CanvasAgentToolNames.requiresCanvas(item.toolName()))
                            .toList(),
                    tools.whitelist().stream().filter(name -> !CanvasAgentToolNames.requiresCanvas(name))
                            .collect(Collectors.toCollection(LinkedHashSet::new)));
        }
        String skillPrompt = skills.root(definition, request, promptVariables, ownerUserId);
        promptVariables.remove("canvasNodeExecution");
        promptVariables.remove("canvasBound");
        if (request.getCanvasContext() != null) {
            promptVariables.put("canvasBound", "true");
        }
        if (canvasNode) {
            promptVariables.put("canvasNodeExecution", "true");
        }
        promptVariables.put(
                TOOL_EXECUTION_MODE_VARIABLE,
                ToolExecutionMode.parse(request.getToolExecutionMode()).name());
        if (ownerUserId != null) {
            promptVariables.put(OWNER_USER_ID_VARIABLE, String.valueOf(ownerUserId));
            String mcpFingerprint = mcpRegistry.userConfigurationFingerprint(ownerUserId);
            if (mcpFingerprint != null && !mcpFingerprint.isBlank()) {
                promptVariables.put(MCP_CONFIG_FINGERPRINT_VARIABLE, mcpFingerprint);
            }
        }
        return create(
                model,
                definitionKey,
                agentName,
                "AgentScope Harness kernel for " + agentName,
                skillPrompt.isEmpty() ? systemPrompt : systemPrompt + "\n\n" + skillPrompt,
                promptVariables,
                tools,
                modelFactory.modelConfigFingerprint(model), definition == null ? null : definition.getMaxIters());
    }

    public AgentKernelSpec createChild(
            AgentKernelSpec parent,
            AiAgentDefinition.SubAgentToolDef subAgent,
            ProjectContext project,
            Map<String, Object> input) {
        Objects.requireNonNull(parent, "parent must not be null");
        Objects.requireNonNull(subAgent, "subAgent must not be null");
        String childType = requireText(subAgent.getRefAgentType(), "subAgent.refAgentType");
        AiAgentDefinition definition = agentService.fromSnapshot(childType, parent.promptVariables());
        var overrides = agentService.snapshotOverride(childType, parent.promptVariables());
        String prompt = overrides.systemPrompt() == null ? normalize(
                subAgent.getSystemPromptOverride()) : definition.getSystemPrompt();
        if (prompt == null) {
            prompt = requireText(definition.getSystemPrompt(), "subAgent.systemPrompt");
        }
        String instruction = overrides.instructionTemplate() == null ? normalize(
                subAgent.getInstructionOverride()) : normalize(definition.getInstructionTemplate());
        if (instruction == null && overrides.instructionTemplate() == null) {
            instruction = normalize(definition.getInstructionTemplate());
        }
        Map<String, String> variables = new LinkedHashMap<>(AgentPromptVariables.forChild(
                parent.promptVariables(), project, input));
        if (parent.promptVariables().containsKey(AgentDefinitionConfiguration.SNAPSHOT_VARIABLE)) {
            variables.put(AgentDefinitionConfiguration.SNAPSHOT_VARIABLE,
                    parent.promptVariables().get(AgentDefinitionConfiguration.SNAPSHOT_VARIABLE));
        }
        prompt = AgentPromptVariables.render(prompt, variables);
        if (instruction != null) {
            prompt = prompt + "\n\n" + AgentPromptVariables.render(instruction, variables);
        }
        String skillPrompt = skills.child(definition, variables, parent.promptVariables());
        if (!skillPrompt.isEmpty()) {
            prompt += "\n\n" + skillPrompt;
        }
        boolean canvasNode = "true".equals(parent.promptVariables().get("canvasNodeExecution"));
        if (canvasNode) {
            var trustedVariables = new LinkedHashMap<>(variables);
            trustedVariables.put("canvasNodeExecution", "true");
            variables = Map.copyOf(trustedVariables);
        }
        ToolSelection tools = selectTools(childType, null,
                canvasNode && Integer.valueOf(1).equals(definition.getEnableTools()) ? List.of() : null,
                ownerUserId(parent), variables);
        if (canvasNode) {
            tools = nodeTools(tools);
        }
        String agentName = overrides.name() == null ? normalize(subAgent.getToolName()) : definition.getName();
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
                parent.key().modelConfigFingerprint(), definition.getMaxIters());
    }

    /**
     * Rebuilds the exact persisted whitelist after verifying every live tool contract.
     */
    public AgentKernelSpec restore(
            AgentKernelSnapshotPayload payload,
            AiModel model,
            String modelFingerprint) {
        Objects.requireNonNull(payload, "payload must not be null");
        Objects.requireNonNull(model, "model must not be null");
        String definitionKey = payload.agentDefinitionStableKey();
        String agentType = DEFAULT_AGENT_KEY.equals(definitionKey) ? null : definitionKey;
        Long ownerUserId = ownerUserId(payload.promptVariables());
        ToolSelection available = selectTools(agentType, null, null, ownerUserId, payload.promptVariables());
        Map<String, AgentKernelToolManifest> availableByName = new LinkedHashMap<>();
        for (AgentKernelToolManifest tool : available.manifest()) {
            availableByName.put(tool.toolName(), tool);
        }
        List<AgentKernelToolManifest> restoredManifest = new ArrayList<>();
        Set<String> restoredWhitelist = new LinkedHashSet<>();
        for (ToolManifestSnapshot persisted : payload.tools()) {
            AgentKernelToolManifest live = availableByName.get(persisted.name());
            if (live == null
                    || !persisted.schemaSha256().equals(live.schemaSha256())
                    || persisted.readOnly() != live.readOnly()
                    || persisted.concurrencySafe() != live.concurrencySafe()
                    || !TOOL_WHITELIST_VERSION.equals(persisted.implementationVersion())) {
                throw new IllegalStateException(
                        "Persisted AgentScope tool is unavailable: " + persisted.name());
            }
            restoredManifest.add(live);
            restoredWhitelist.add(live.toolName());
        }
        AgentKernelKey key = AgentKernelKey.create(
                definitionKey,
                modelFingerprint,
                AgentKernelKey.promptVersion(
                        payload.systemPrompt(), payload.promptVariables()),
                restoredManifest,
                TOOL_WHITELIST_VERSION);
        return new AgentKernelSpec(
                key,
                model,
                definitionKey,
                payload.agentName(),
                payload.description(),
                payload.systemPrompt(),
                payload.promptVariables(),
                payload.maxIters(),
                restoredManifest,
                restoredWhitelist,
                TOOL_WHITELIST_VERSION);
    }

    private AgentKernelSpec create(
            AiModel model,
            String definitionKey,
            String agentName,
            String description,
            String systemPrompt,
            Map<String, String> promptVariables,
            ToolSelection tools,
            String modelFingerprint,
            Integer maxIters) {
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
                AgentExecutionLimits.frameworkSteps(
                        maxIters == null ? properties.getExecution().getMaxIters() : maxIters),
                tools.manifest(),
                tools.whitelist(),
                TOOL_WHITELIST_VERSION);
    }

    private ToolSelection nodeTools(ToolSelection tools) {
        // 专业节点只产生输入范围内的业务结果；图编辑和嵌套工作流由画布助手负责。
        return new ToolSelection(
                tools.manifest().stream().filter(item -> CanvasAgentNodeTypes.permittedTool(item.toolName())).toList(),
                tools.whitelist().stream().filter(CanvasAgentNodeTypes::permittedTool)
                        .collect(Collectors.toCollection(LinkedHashSet::new)));
    }

    private ToolSelection selectTools(
            String agentType,
            List<String> requestedTools,
            List<String> requestedMcpTools,
            Long ownerUserId,
            Map<String, String> variables) {
        Set<String> requested = requestedTools == null || requestedTools.isEmpty()
                ? null
                : Set.copyOf(requestedTools);
        Set<String> requestedMcp = requestedMcpTools == null
                ? null
                : Set.copyOf(requestedMcpTools);
        AiAgentDefinition definition = agentType == null ? null : agentService.fromSnapshot(agentType, variables);
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
                : toolConfigService.getEnabledToolsByDefinition(definition);
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
                : definition.getSubAgentTools();
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
        List<AgentKernelToolManifest> availableMcpTools = ownerUserId == null
                ? mcpRegistry.manifestsForAgent(definitionKey)
                : mcpRegistry.manifestsForAgent(definitionKey, ownerUserId);
        if (definition != null && definition.getMcpToolNames() != null) {
            Set<String> configured = Set.copyOf(definition.getMcpToolNames());
            availableMcpTools = availableMcpTools.stream().filter(tool -> configured.contains(tool.toolName()))
                    .toList();
            Set<String> discovered = availableMcpTools.stream().map(AgentKernelToolManifest::toolName)
                    .collect(Collectors.toSet());
            if (!discovered.equals(configured)) {
                throw new IllegalArgumentException("智能体配置中的 MCP 工具已不可用，请更新配置");
            }
        }
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

    public static Long ownerUserId(AgentKernelSpec spec) {
        return ownerUserId(spec.promptVariables());
    }

    private static Long ownerUserId(Map<String, String> promptVariables) {
        String value = promptVariables.get(OWNER_USER_ID_VARIABLE);
        if (value == null || value.isBlank()) {
            return null;
        }
        try {
            return Long.valueOf(value);
        } catch (NumberFormatException failure) {
            throw new IllegalArgumentException("Invalid AgentScope kernel owner user id", failure);
        }
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
