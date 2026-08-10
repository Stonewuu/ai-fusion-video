package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.stonewu.fusion.config.ai.AiAgentDefinition;
import com.stonewu.fusion.service.ai.AiAgentService;
import com.stonewu.fusion.service.ai.AiToolConfigService;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.ToolExecutorRegistry;
import com.stonewu.fusion.service.ai.agentscope.AgentScopeSubAgentToolAdapter;
import com.stonewu.fusion.service.ai.agentscope.AgentScopeToolAdapter;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentScopeMcpRegistry;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRunPort;
import com.stonewu.fusion.service.ai.run.RunLeaseGuard;
import io.agentscope.core.tool.AgentTool;
import io.agentscope.core.tool.ToolBase;
import io.agentscope.core.tool.Toolkit;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Registers only the immutable tool whitelist captured by an AgentScope Harness kernel. */
@Component
public final class PlatformAgentKernelToolRegistry implements AgentKernelToolRegistry {

    private final ToolExecutorRegistry executors;
    private final AiToolConfigService toolConfigService;
    private final AiAgentService agentService;
    private final AgentKernelSpecFactory specFactory;
    private final ObjectProvider<PlatformSubAgentRunPort> childRuns;
    private final AgentRuntimeSchedulers schedulers;
    private final RunLeaseGuard leaseGuard;
    private final ObjectMapper objectMapper;
    private final AgentScopeMcpRegistry mcpRegistry;

    public PlatformAgentKernelToolRegistry(
            ToolExecutorRegistry executors,
            AiToolConfigService toolConfigService,
            AiAgentService agentService,
            AgentKernelSpecFactory specFactory,
            ObjectProvider<PlatformSubAgentRunPort> childRuns,
            AgentRuntimeSchedulers schedulers,
            RunLeaseGuard leaseGuard,
            ObjectMapper objectMapper,
            AgentScopeMcpRegistry mcpRegistry) {
        this.executors = Objects.requireNonNull(executors, "executors must not be null");
        this.toolConfigService = Objects.requireNonNull(
                toolConfigService, "toolConfigService must not be null");
        this.agentService = Objects.requireNonNull(agentService, "agentService must not be null");
        this.specFactory = Objects.requireNonNull(specFactory, "specFactory must not be null");
        this.childRuns = Objects.requireNonNull(childRuns, "childRuns must not be null");
        this.schedulers = Objects.requireNonNull(schedulers, "schedulers must not be null");
        this.leaseGuard = Objects.requireNonNull(leaseGuard, "leaseGuard must not be null");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
        this.mcpRegistry = Objects.requireNonNull(mcpRegistry, "mcpRegistry must not be null");
    }

    @Override
    public AgentKernelToolkitResources register(AgentKernelSpec spec, Toolkit toolkit) {
        Objects.requireNonNull(spec, "spec must not be null");
        Objects.requireNonNull(toolkit, "toolkit must not be null");
        Map<String, ToolExecutor> direct = directTools(spec.agentDefinitionStableKey());
        Map<String, AiAgentDefinition.SubAgentToolDef> children =
                subAgentTools(spec.agentDefinitionStableKey());
        Map<String, AgentKernelToolManifest> manifest = manifest(spec);
        Long ownerUserId = AgentKernelSpecFactory.ownerUserId(spec);
        AgentKernelToolkitResources mcpResources = AgentKernelToolkitResources.none();
        if (ownerUserId == null) {
            mcpRegistry.register(
                    spec.agentDefinitionStableKey(), spec.toolWhitelist(), toolkit);
        } else {
            mcpResources = mcpRegistry.register(
                    spec.agentDefinitionStableKey(), ownerUserId, spec.toolWhitelist(), toolkit);
        }

        try {
            for (String toolName : spec.toolWhitelist()) {
                ToolExecutor executor = direct.get(toolName);
                AiAgentDefinition.SubAgentToolDef child = children.get(toolName);
                boolean mcpTool = ownerUserId == null
                        ? mcpRegistry.isMcpTool(toolName, spec.agentDefinitionStableKey())
                        : mcpRegistry.isMcpTool(
                                toolName, spec.agentDefinitionStableKey(), ownerUserId);
                int matches = (executor == null ? 0 : 1)
                        + (child == null ? 0 : 1)
                        + (mcpTool ? 1 : 0);
                if (matches > 1) {
                    throw new IllegalStateException(
                            "AgentScope tool name is ambiguous in kernel whitelist: " + toolName);
                }
                AgentKernelToolManifest expected = Objects.requireNonNull(
                        manifest.get(toolName), "Missing kernel tool manifest: " + toolName);
                if (executor != null) {
                    AgentScopeToolSchema.PreparedSchema schema = AgentScopeToolSchema.prepare(
                            objectMapper, executor.getParametersSchema(), toolName);
                    requireManifest(
                            expected, schema, executor.isReadOnly(), executor.isConcurrencySafe());
                    toolkit.registerAgentTool(new AgentScopeToolAdapter(
                            executor, schema, schedulers.toolBlocking(), leaseGuard, objectMapper));
                } else if (child != null) {
                    AgentScopeToolSchema.PreparedSchema schema =
                            AgentScopeToolSchema.prepareSubAgent(
                                    objectMapper, child.getParametersSchema(), toolName);
                    requireManifest(expected, schema, false, true);
                    toolkit.registerAgentTool(new AgentScopeSubAgentToolAdapter(
                            child,
                            spec,
                            schema,
                            specFactory,
                            childRuns::getIfAvailable,
                            leaseGuard,
                            objectMapper));
                } else if (mcpTool) {
                    AgentTool registered = Objects.requireNonNull(
                            toolkit.getTool(toolName),
                            "MCP registry did not register tool: " + toolName);
                    AgentScopeToolSchema.PreparedSchema schema = prepareMcpSchema(
                            registered, toolName);
                    boolean concurrencySafe = registered instanceof ToolBase toolBase
                            && toolBase.isConcurrencySafe();
                    requireManifest(
                            expected,
                            schema,
                            registered.isReadOnly(),
                            concurrencySafe);
                } else {
                    throw new IllegalStateException(
                            "AgentScope kernel references an unavailable platform tool: " + toolName);
                }
            }
            return mcpResources;
        } catch (Throwable failure) {
            try {
                mcpResources.close();
            } catch (Throwable closeFailure) {
                failure.addSuppressed(closeFailure);
            }
            if (failure instanceof RuntimeException runtimeFailure) {
                throw runtimeFailure;
            }
            if (failure instanceof Error error) {
                throw error;
            }
            throw new IllegalStateException("AgentScope tool registration failed", failure);
        }
    }

    private AgentScopeToolSchema.PreparedSchema prepareMcpSchema(
            AgentTool tool,
            String toolName) {
        try {
            return AgentScopeToolSchema.prepare(
                    objectMapper,
                    objectMapper.writeValueAsString(tool.getParameters()),
                    toolName);
        } catch (JsonProcessingException failure) {
            throw new IllegalStateException(
                    "Failed to canonicalize registered MCP tool schema: " + toolName,
                    failure);
        }
    }

    private Map<String, ToolExecutor> directTools(String definitionKey) {
        List<ToolExecutor> allowed = AgentKernelSpecFactory.DEFAULT_AGENT_KEY.equals(definitionKey)
                ? toolConfigService.getEnabledTools()
                : toolConfigService.getEnabledToolsByAgent(definitionKey);
        Map<String, ToolExecutor> result = new LinkedHashMap<>();
        for (ToolExecutor tool : allowed) {
            ToolExecutor registered = executors.findExecutor(tool.getToolName());
            if (registered == null || registered != tool) {
                throw new IllegalStateException(
                        "Platform tool registry identity changed: " + tool.getToolName());
            }
            if (result.put(tool.getToolName(), tool) != null) {
                throw new IllegalStateException("Duplicate platform tool: " + tool.getToolName());
            }
        }
        return result;
    }

    private Map<String, AiAgentDefinition.SubAgentToolDef> subAgentTools(String definitionKey) {
        if (AgentKernelSpecFactory.DEFAULT_AGENT_KEY.equals(definitionKey)) {
            return Map.of();
        }
        agentService.getRequiredByType(definitionKey);
        Map<String, AiAgentDefinition.SubAgentToolDef> result = new LinkedHashMap<>();
        for (AiAgentDefinition.SubAgentToolDef child
                : toolConfigService.getSubAgentTools(definitionKey)) {
            if (result.put(child.getToolName(), child) != null) {
                throw new IllegalStateException(
                        "Duplicate platform sub-agent tool: " + child.getToolName());
            }
        }
        return result;
    }

    private Map<String, AgentKernelToolManifest> manifest(AgentKernelSpec spec) {
        Map<String, AgentKernelToolManifest> result = new LinkedHashMap<>();
        for (AgentKernelToolManifest entry : spec.toolManifest()) {
            result.put(entry.toolName(), entry);
        }
        return result;
    }

    private void requireManifest(
            AgentKernelToolManifest expected,
            AgentScopeToolSchema.PreparedSchema schema,
            boolean readOnly,
            boolean concurrencySafe) {
        String actualHash = AgentKernelToolManifest.schemaSha256(schema.canonicalJson());
        if (!expected.schemaSha256().equals(actualHash)
                || expected.readOnly() != readOnly
                || expected.concurrencySafe() != concurrencySafe) {
            throw new IllegalStateException(
                    "Platform AgentScope tool contract changed: " + expected.toolName());
        }
    }
}
