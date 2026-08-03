package com.stonewu.fusion.service.ai.agentscope.mcp;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelToolManifest;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelToolkitResources;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import io.agentscope.core.tool.Toolkit;
import io.agentscope.core.tool.mcp.McpClientBuilder;
import io.agentscope.core.tool.mcp.McpClientWrapper;
import io.agentscope.core.tool.mcp.McpTool;
import io.modelcontextprotocol.spec.McpSchema;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Owns application-level MCP clients and exposes their discovered tools to immutable AgentScope
 * kernels. Clients are shared between cached kernels and closed once during application shutdown.
 */
@Component
public final class AgentScopeMcpRegistry implements DisposableBean {

    private static final Logger log = LoggerFactory.getLogger(AgentScopeMcpRegistry.class);

    private final ObjectMapper objectMapper;
    private final Map<String, RegisteredServer> servers;
    private final Map<String, McpToolDescriptor> tools;
    private final AgentUserMcpRuntimeRegistry userRegistry;

    @Autowired
    public AgentScopeMcpRegistry(
            AgentScopeV2Properties properties,
            ObjectMapper objectMapper,
            ObjectProvider<AgentUserMcpRuntimeRegistry> userRegistries) {
        this(properties, objectMapper, AgentScopeMcpRegistry::createClient,
                userRegistries.getIfAvailable());
    }

    public AgentScopeMcpRegistry(
            AgentScopeV2Properties properties,
            ObjectMapper objectMapper) {
        this(properties, objectMapper, AgentScopeMcpRegistry::createClient, null);
    }

    AgentScopeMcpRegistry(
            AgentScopeV2Properties properties,
            ObjectMapper objectMapper,
            McpClientFactory clientFactory) {
        this(properties, objectMapper, clientFactory, null);
    }

    private AgentScopeMcpRegistry(
            AgentScopeV2Properties properties,
            ObjectMapper objectMapper,
            McpClientFactory clientFactory,
            AgentUserMcpRuntimeRegistry userRegistry) {
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
        this.userRegistry = userRegistry;
        AgentScopeV2Properties.Mcp config = Objects.requireNonNull(
                properties, "properties must not be null").getMcp();
        RegistrySnapshot snapshot = config.isEnabled()
                ? connect(config, Objects.requireNonNull(clientFactory, "clientFactory must not be null"))
                : RegistrySnapshot.empty();
        this.servers = snapshot.servers();
        this.tools = snapshot.tools();
        log.info("AgentScope MCP initialized [enabled={}, servers={}, tools={}]",
                !servers.isEmpty(), servers.size(), tools.size());
    }

    public List<AgentKernelToolManifest> manifestsForAgent(String agentDefinitionStableKey) {
        List<AgentKernelToolManifest> manifests = new ArrayList<>();
        for (McpToolDescriptor descriptor : tools.values()) {
            if (descriptor.availableTo(agentDefinitionStableKey)) {
                manifests.add(descriptor.manifest());
            }
        }
        return List.copyOf(manifests);
    }

    public List<AgentKernelToolManifest> manifestsForAgent(
            String agentDefinitionStableKey,
            Long userId) {
        List<AgentKernelToolManifest> manifests = new ArrayList<>(
                manifestsForAgent(agentDefinitionStableKey));
        if (userRegistry == null || userId == null) {
            return List.copyOf(manifests);
        }
        Set<String> names = new LinkedHashSet<>();
        manifests.forEach(manifest -> names.add(manifest.toolName()));
        for (AgentKernelToolManifest manifest : userRegistry.manifests(userId)) {
            if (!names.add(manifest.toolName())) {
                throw new IllegalArgumentException(
                        "用户 MCP 工具与系统 MCP 工具重名: " + manifest.toolName());
            }
            manifests.add(manifest);
        }
        return List.copyOf(manifests);
    }

    public boolean isMcpTool(String toolName, String agentDefinitionStableKey) {
        McpToolDescriptor descriptor = tools.get(toolName);
        return descriptor != null && descriptor.availableTo(agentDefinitionStableKey);
    }

    public boolean isMcpTool(
            String toolName,
            String agentDefinitionStableKey,
            Long userId) {
        return isMcpTool(toolName, agentDefinitionStableKey)
                || (userRegistry != null && userId != null && userRegistry.isTool(userId, toolName));
    }

    public List<McpToolReference> catalogForAgent(String agentDefinitionStableKey) {
        List<McpToolReference> catalog = new ArrayList<>();
        for (Map.Entry<String, McpToolDescriptor> entry : tools.entrySet()) {
            McpToolDescriptor descriptor = entry.getValue();
            if (descriptor.availableTo(agentDefinitionStableKey)) {
                catalog.add(new McpToolReference(
                        descriptor.serverName(),
                        entry.getKey(),
                        descriptor.description(),
                        descriptor.manifest().readOnly()));
            }
        }
        catalog.sort(Comparator
                .comparing(McpToolReference::serverName)
                .thenComparing(McpToolReference::toolName));
        return List.copyOf(catalog);
    }

    public List<McpToolReference> catalogForAgent(
            String agentDefinitionStableKey,
            Long userId) {
        List<McpToolReference> catalog = new ArrayList<>(catalogForAgent(agentDefinitionStableKey));
        if (userRegistry != null && userId != null) {
            Set<String> names = new LinkedHashSet<>();
            catalog.forEach(tool -> names.add(tool.toolName()));
            for (McpToolReference tool : userRegistry.catalog(userId)) {
                if (!names.add(tool.toolName())) {
                    throw new IllegalArgumentException(
                            "用户 MCP 工具与系统 MCP 工具重名: " + tool.toolName());
                }
                catalog.add(tool);
            }
            catalog.sort(Comparator
                    .comparing(McpToolReference::serverName)
                    .thenComparing(McpToolReference::toolName));
        }
        return List.copyOf(catalog);
    }

    /** Registers only the MCP tools already captured by the kernel whitelist. */
    public void register(
            String agentDefinitionStableKey,
            Set<String> kernelWhitelist,
            Toolkit toolkit) {
        Objects.requireNonNull(kernelWhitelist, "kernelWhitelist must not be null");
        Objects.requireNonNull(toolkit, "toolkit must not be null");
        Map<String, List<String>> selectedByServer = new LinkedHashMap<>();
        for (String toolName : kernelWhitelist) {
            McpToolDescriptor descriptor = tools.get(toolName);
            if (descriptor == null || !descriptor.availableTo(agentDefinitionStableKey)) {
                continue;
            }
            selectedByServer.computeIfAbsent(descriptor.serverName(), ignored -> new ArrayList<>())
                    .add(toolName);
        }
        for (Map.Entry<String, List<String>> selection : selectedByServer.entrySet()) {
            RegisteredServer server = Objects.requireNonNull(
                    servers.get(selection.getKey()), "MCP server disappeared from registry");
            toolkit.registration()
                    .mcpClient(server.client())
                    .enableTools(selection.getValue())
                    .apply();
        }
        for (List<String> names : selectedByServer.values()) {
            for (String name : names) {
                if (toolkit.getTool(name) == null) {
                    throw new IllegalStateException(
                            "Configured MCP tool was not registered by AgentScope: " + name);
                }
            }
        }
    }

    public AgentKernelToolkitResources register(
            String agentDefinitionStableKey,
            Long userId,
            Set<String> kernelWhitelist,
            Toolkit toolkit) {
        register(agentDefinitionStableKey, kernelWhitelist, toolkit);
        if (userRegistry != null && userId != null) {
            return userRegistry.register(userId, kernelWhitelist, toolkit);
        }
        return AgentKernelToolkitResources.none();
    }

    public String userConfigurationFingerprint(Long userId) {
        return userRegistry == null || userId == null
                ? AgentKernelToolManifest.schemaSha256("")
                : userRegistry.configurationFingerprint(userId);
    }

    @Override
    public void destroy() {
        RuntimeException failure = null;
        for (RegisteredServer server : servers.values()) {
            try {
                server.client().close();
            } catch (RuntimeException closeFailure) {
                if (failure == null) {
                    failure = new IllegalStateException("Failed to close AgentScope MCP clients");
                }
                failure.addSuppressed(closeFailure);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    private RegistrySnapshot connect(
            AgentScopeV2Properties.Mcp config,
            McpClientFactory clientFactory) {
        Map<String, RegisteredServer> connectedServers = new LinkedHashMap<>();
        Map<String, McpToolDescriptor> discoveredTools = new LinkedHashMap<>();
        for (Map.Entry<String, AgentScopeV2Properties.McpServer> entry
                : config.getServers().entrySet()) {
            String serverName = requireText(entry.getKey(), "MCP server name");
            AgentScopeV2Properties.McpServer serverConfig = entry.getValue();
            if (serverConfig == null || !serverConfig.isEnabled()) {
                continue;
            }
            McpClientWrapper client = null;
            try {
                client = Objects.requireNonNull(
                        clientFactory.create(serverName, serverConfig),
                        "MCP client factory returned null");
                List<McpSchema.Tool> discovered = discoverTools(client, serverConfig);
                RegisteredServer server = describeServer(
                        serverName, serverConfig, client, discovered, discoveredTools);
                connectedServers.put(serverName, server);
                client = null;
            } catch (Exception failure) {
                closeQuietly(client);
                if (config.isFailFast()) {
                    closeQuietly(connectedServers.values());
                    throw new IllegalStateException(
                            "Failed to initialize AgentScope MCP server: " + serverName,
                            failure);
                }
                log.warn("Skipping unavailable AgentScope MCP server '{}': {}",
                        serverName, failure.getMessage());
            }
        }
        return new RegistrySnapshot(
                Map.copyOf(connectedServers),
                Map.copyOf(discoveredTools));
    }

    private List<McpSchema.Tool> discoverTools(
            McpClientWrapper client,
            AgentScopeV2Properties.McpServer config) {
        Duration wait = config.getInitializationTimeout().plus(config.getTimeout());
        List<McpSchema.Tool> discovered = client.initialize()
                .then(client.listTools())
                .block(wait);
        if (discovered == null) {
            throw new IllegalStateException("MCP server returned no tool catalog");
        }
        return List.copyOf(discovered);
    }

    private RegisteredServer describeServer(
            String serverName,
            AgentScopeV2Properties.McpServer config,
            McpClientWrapper client,
            List<McpSchema.Tool> discovered,
            Map<String, McpToolDescriptor> allTools) {
        Map<String, McpSchema.Tool> byName = new LinkedHashMap<>();
        for (McpSchema.Tool tool : discovered) {
            String toolName = requireText(tool.name(), "MCP tool name");
            if (byName.put(toolName, tool) != null) {
                throw new IllegalStateException(
                        "MCP server exposes duplicate tool name: " + toolName);
            }
        }
        Set<String> enabledNames = config.getEnabledTools().isEmpty()
                ? new LinkedHashSet<>(byName.keySet())
                : new LinkedHashSet<>(config.getEnabledTools());
        Set<String> unavailable = new LinkedHashSet<>(enabledNames);
        unavailable.removeAll(byName.keySet());
        if (!unavailable.isEmpty()) {
            throw new IllegalStateException(
                    "Configured MCP tools are unavailable from " + serverName + ": " + unavailable);
        }
        Set<String> duplicates = new LinkedHashSet<>(enabledNames);
        duplicates.retainAll(allTools.keySet());
        if (!duplicates.isEmpty()) {
            throw new IllegalStateException(
                    "MCP tool names are exposed by multiple servers: " + duplicates);
        }

        Map<String, McpToolDescriptor> serverTools = new LinkedHashMap<>();
        Set<String> agentTypes = Set.copyOf(config.getAgentTypes());
        for (String toolName : enabledNames) {
            McpSchema.Tool tool = byName.get(toolName);
            AgentKernelToolManifest manifest = manifest(tool);
            McpToolDescriptor descriptor = new McpToolDescriptor(
                    serverName,
                    tool.description() == null ? "" : tool.description(),
                    manifest,
                    agentTypes);
            serverTools.put(toolName, descriptor);
        }
        allTools.putAll(serverTools);
        return new RegisteredServer(client);
    }

    private AgentKernelToolManifest manifest(McpSchema.Tool tool) {
        Map<String, Object> parameters = McpTool.convertMcpSchemaToParameters(
                tool.inputSchema(), Set.of());
        try {
            AgentScopeToolSchema.PreparedSchema schema = AgentScopeToolSchema.prepare(
                    objectMapper,
                    objectMapper.writeValueAsString(parameters),
                    tool.name());
            boolean readOnly = tool.annotations() != null
                    && Boolean.TRUE.equals(tool.annotations().readOnlyHint());
            return new AgentKernelToolManifest(
                    tool.name(),
                    AgentKernelToolManifest.schemaSha256(schema.canonicalJson()),
                    readOnly,
                    false);
        } catch (JsonProcessingException serializationFailure) {
            throw new IllegalStateException(
                    "Failed to canonicalize MCP tool schema: " + tool.name(),
                    serializationFailure);
        }
    }

    private static McpClientWrapper createClient(
            String name,
            AgentScopeV2Properties.McpServer config) {
        McpClientBuilder builder = McpClientBuilder.create(name);
        String transport = requireText(config.getTransport(), "MCP transport")
                .toLowerCase(Locale.ROOT);
        switch (transport) {
            case "stdio" -> builder.stdioTransport(
                    requireText(config.getCommand(), "stdio MCP command"),
                    config.getArgs(),
                    config.getEnv());
            case "sse" -> configureHttp(builder.sseTransport(
                    requireText(config.getUrl(), "SSE MCP URL")), config);
            case "http", "streamable-http", "streamablehttp" -> configureHttp(
                    builder.streamableHttpTransport(
                            requireText(config.getUrl(), "HTTP MCP URL")),
                    config);
            default -> throw new IllegalArgumentException(
                    "Unsupported MCP transport: " + config.getTransport());
        }
        if (!config.getProtocolVersions().isEmpty()) {
            builder.protocolVersions(config.getProtocolVersions().toArray(String[]::new));
        }
        return builder.timeout(config.getTimeout())
                .initializationTimeout(config.getInitializationTimeout())
                .buildAsync()
                .block(config.getInitializationTimeout());
    }

    private static void configureHttp(
            McpClientBuilder builder,
            AgentScopeV2Properties.McpServer config) {
        if (!config.getHeaders().isEmpty()) {
            builder.headers(config.getHeaders());
        }
        if (!config.getQueryParams().isEmpty()) {
            builder.queryParams(config.getQueryParams());
        }
    }

    private static void closeQuietly(McpClientWrapper client) {
        if (client == null) {
            return;
        }
        try {
            client.close();
        } catch (RuntimeException ignored) {
            // Preserve the initialization failure as the primary cause.
        }
    }

    private static void closeQuietly(Iterable<RegisteredServer> registeredServers) {
        for (RegisteredServer server : registeredServers) {
            closeQuietly(server.client());
        }
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value.trim();
    }

    @FunctionalInterface
    interface McpClientFactory {
        McpClientWrapper create(String name, AgentScopeV2Properties.McpServer config);
    }

    private record RegisteredServer(McpClientWrapper client) {
    }

    private record McpToolDescriptor(
            String serverName,
            String description,
            AgentKernelToolManifest manifest,
            Set<String> agentTypes) {

        private boolean availableTo(String agentDefinitionStableKey) {
            return agentTypes.isEmpty() || agentTypes.contains(agentDefinitionStableKey);
        }
    }

    public record McpToolReference(
            String serverName,
            String toolName,
            String description,
            boolean readOnly) {
    }

    private record RegistrySnapshot(
            Map<String, RegisteredServer> servers,
            Map<String, McpToolDescriptor> tools) {

        private static RegistrySnapshot empty() {
            return new RegistrySnapshot(Map.of(), Map.of());
        }
    }
}
